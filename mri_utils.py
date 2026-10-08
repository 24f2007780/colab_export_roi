"""Notebook helpers for stroke MRI NIfTI files.

Needs only nibabel, numpy, pandas, scipy, scikit-image and matplotlib (see requirements.txt).

Conventions
- Data arrays are float32 with the NIfTI slope/intercept already applied.
- World coordinates come from the NIfTI affine, RAS+ in mm: +x points to patient Right,
  so x < 0 is LEFT; +y is anterior, so y < 0 is posterior; +z is superior.
- Lesion thresholds are never chosen for you. Pass them explicitly, e.g. threshold=150.
  Direction: "above" (hyperintense, typical DWI) or "below" (hypointense, typical ADC).
  The default "auto" uses "below" for ADC files and "above" otherwise, and prints the rule used.
- Section 3 functions take an already-loaded `manifest` dict (json.load(open("manifest.json"))) to bridge
  to the viewer's histology block/atlas geometry. They only ever touch the manifest dict and paths passed to
  them explicitly, and they never assume a NIfTI file's array axis order matches manifest["space"]["axes"]
  (i, j, k) - see _align_to_manifest_shape.
"""
from dataclasses import dataclass
from pathlib import Path
import gzip
import json
import re

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage as ndi, stats
from skimage.filters import threshold_otsu

__all__ = ["MRI", "load_mri", "load_manifest", "inspect_mri", "intensity_qc", "plot_intensity_qc",
           "brain_mask", "mask_stats", "qc_summary", "lesion_mask",
           "lesion_stats", "threshold_sensitivity", "show_mri",
           "check_spatial_alignment", "DiffusionPair", "load_diffusion_pair",
           "paired_diffusion_stats", "restricted_diffusion_lesion",
           "paired_threshold_sensitivity", "plot_joint_diffusion", "show_diffusion_pair",
           "compare_mri_volumes", "load_manifest_volume", "regional_lesion_burden",
           "map_voxel_to_histology_section", "extract_block_plane_from_mri",
           "lesion_border_profile"]

_PLANE_WORLD_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}   # plane is perpendicular to this world axis


@dataclass
class MRI:
    """A loaded NIfTI volume: scaled data plus the header-level geometry needed for mm work."""
    path: Path
    data: np.ndarray          # float32, NIfTI scaling applied
    img: nib.Nifti1Image

    @property
    def affine(self):
        return self.img.affine

    @property
    def shape(self):
        return self.data.shape

    @property
    def voxel_mm(self):
        return tuple(float(v) for v in self.img.header.get_zooms()[:3])

    @property
    def raw_dtype(self):
        return str(self.img.get_data_dtype())

    @property
    def orientation(self):
        return "".join(nib.aff2axcodes(self.affine))

    @property
    def origin_mm(self):
        return tuple(float(v) for v in self.affine[:3, 3])

    @property
    def extent_mm(self):
        """Physical bounding box ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in world mm."""
        corners = np.array([
            [0, 0, 0, 1],
            [self.shape[0] - 1, 0, 0, 1],
            [0, self.shape[1] - 1, 0, 1],
            [self.shape[0] - 1, self.shape[1] - 1, 0, 1],
            [0, 0, self.shape[2] - 1, 1],
            [self.shape[0] - 1, 0, self.shape[2] - 1, 1],
            [0, self.shape[1] - 1, self.shape[2] - 1, 1],
            [self.shape[0] - 1, self.shape[1] - 1, self.shape[2] - 1, 1],
        ], dtype=float)
        wc = (self.affine @ corners.T).T[:, :3]
        mn = wc.min(axis=0)
        mx = wc.max(axis=0)
        return ((float(mn[0]), float(mx[0])), (float(mn[1]), float(mx[1])), (float(mn[2]), float(mx[2])))

    @property
    def fov_mm(self):
        """Field of view (span along x, y, z in world mm)."""
        ext = self.extent_mm
        return (float(ext[0][1] - ext[0][0]), float(ext[1][1] - ext[1][0]), float(ext[2][1] - ext[2][0]))

    @property
    def modality(self):
        name = self.path.name
        if re.search(r"ADC", name, re.I):
            return "ADC"
        if re.search(r"DWI", name, re.I):
            return "DWI"
        return "unknown"


def _resolve_data_path(path):
    """Resolve a file path, checking current directory, data/, ../data/, and project root."""
    p = Path(path)
    if p.exists():
        return p
    candidates = [
        Path("data") / p,
        Path("../data") / p,
        Path("..") / p,
        Path(__file__).resolve().parent / "data" / p,
        Path(__file__).resolve().parent / p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return p


def load_manifest(path="manifest.json"):
    """Load manifest.json from local path, data/, or repository root."""
    path = _resolve_data_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mri(path):
    """Load a NIfTI file (.nii or .nii.gz) as float32 with slope/intercept applied.

    Searches current working directory, data/, ../data/, and project root automatically.
    get_fdata applies scaling. np.asanyarray(img.dataobj) would give raw integers for a scaled file.
    """
    path = _resolve_data_path(path)
    img = nib.load(str(path))
    return MRI(path=path, data=img.get_fdata(dtype=np.float32), img=img)


def inspect_mri(mri):
    """Return a one-row-per-field summary of the file's geometry and intensities."""
    A = mri.affine[:3, :3]
    cols = A / np.linalg.norm(A, axis=0)
    # angle between each voxel axis and its nearest world axis; 0 means axis-aligned
    tilt = np.degrees(np.arccos(np.clip(np.abs(cols).max(axis=0), 0, 1)))
    hdr = mri.img.header
    finite = mri.data[np.isfinite(mri.data)]
    ext = mri.extent_mm
    fov = mri.fov_mm
    rows = {
        "file": mri.path.name,
        "modality (from filename)": mri.modality,
        "shape (voxels)": str(mri.shape),
        "raw / loaded dtype": f"{mri.raw_dtype} / {mri.data.dtype}",
        "voxel size (mm)": str(tuple(round(v, 3) for v in mri.voxel_mm)),
        "orientation (axcodes)": mri.orientation,
        "physical extent (mm)": f"X[{ext[0][0]:.1f}, {ext[0][1]:.1f}] Y[{ext[1][0]:.1f}, {ext[1][1]:.1f}] Z[{ext[2][0]:.1f}, {ext[2][1]:.1f}]",
        "field of view (mm)": str(tuple(round(v, 1) for v in fov)),
        "max voxel-axis tilt (deg)": round(float(tilt.max()), 2),
        "sform code": int(hdr["sform_code"]),
        "qform code": int(hdr["qform_code"]),
        "scl slope / intercept": f"{hdr['scl_slope']} / {hdr['scl_inter']}",
        "intensity min / median / max": f"{finite.min():.4g} / {np.median(finite):.4g} / {finite.max():.4g}",
        "zero voxels (after scaling)": f"{(mri.data == 0).mean():.1%}",
        "description": hdr["descrip"].tobytes().rstrip(b"\x00").decode(errors="ignore") or "(none)",
        "b-values / gradients": "not in file",
    }
    return pd.DataFrame({"value": rows})


def intensity_qc(mri, mask=None):
    """Detailed intensity quality control and anomaly detection.

    Detects:
    - Background zero fraction and intercept shifts (e.g. scl_inter turning raw zeros to small floats)
    - Upper saturation / clipping (e.g. uint8 ceiling at 255 scaled)
    - Robust percentiles and distribution metrics.
    """
    d = mri.data[mask] if mask is not None else mri.data
    finite = d[np.isfinite(d)]
    hdr = mri.img.header
    dmin = float(finite.min())
    dmax = float(finite.max())
    n_total = finite.size

    zero_count = int((finite == 0).sum())
    near_min_count = int(np.isclose(finite, dmin, atol=1e-4).sum())
    near_max_count = int(np.isclose(finite, dmax, atol=1e-4).sum())

    q = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])

    # Ceiling clipping flag: > 0.05% voxels at ceiling and max matches uint8 range or near ceiling
    is_clipped = bool((near_max_count / n_total > 0.0005) and (mri.raw_dtype == "uint8" or dmax in (255, 258.5222, 196.8343)))
    # Background trap flag: min > 0 and > 10% voxels are clustered at min value
    is_bg_trap = bool((dmin > 0) and (near_min_count / n_total > 0.10))

    return {
        "file": mri.path.name,
        "modality": mri.modality,
        "min": dmin,
        "p1": float(q[0]),
        "p25": float(q[2]),
        "median": float(q[3]),
        "p75": float(q[4]),
        "p99": float(q[6]),
        "max": dmax,
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "zero_pct": round(zero_count / n_total, 4),
        "near_min_pct": round(near_min_count / n_total, 4),
        "ceiling_pct": round(near_max_count / n_total, 4),
        "bg_trap_flag": is_bg_trap,
        "clipping_flag": is_clipped,
    }


def plot_intensity_qc(mri, mask=None, bins=100, figsize=(13, 3.5)):
    """Plot visual quality control histograms and cumulative distribution for a volume."""
    d = mri.data[mask] if mask is not None else mri.data
    finite = d[np.isfinite(d)]
    qc = intensity_qc(mri, mask=mask)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Full-range log histogram
    axes[0].hist(finite, bins=bins, color="teal", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{mri.path.name}\nFull Volume (Log scale)")
    axes[0].set_xlabel("Voxel Intensity")
    axes[0].set_ylabel("Count (log)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    if qc["bg_trap_flag"]:
        axes[0].axvline(qc["min"], color="orange", linestyle="--", label=f"Min floor ({qc['min']:.4g})")
    if qc["clipping_flag"]:
        axes[0].axvline(qc["max"], color="crimson", linestyle="--", label=f"Ceiling ({qc['max']:.4g})")
    if qc["bg_trap_flag"] or qc["clipping_flag"]:
        axes[0].legend(fontsize=8)

    # Panel 2: Foreground/Tissue distribution (excluding background floor)
    fg = finite[finite > (qc["min"] + 1e-4)] if qc["bg_trap_flag"] else finite[finite > 0]
    if len(fg) > 0:
        axes[1].hist(fg, bins=bins, color="purple", alpha=0.7)
        axes[1].axvline(qc["median"], color="black", linestyle="-", label=f"Median ({qc['median']:.1f})")
        axes[1].axvline(qc["p25"], color="gray", linestyle=":", label=f"IQR ({qc['p25']:.1f} - {qc['p75']:.1f})")
        axes[1].axvline(qc["p75"], color="gray", linestyle=":")
        axes[1].set_title("Tissue Range (Excl. Background Floor)")
        axes[1].set_xlabel("Voxel Intensity")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Empirical cumulative distribution function (CDF)
    sorted_vals = np.sort(finite[::max(1, len(finite) // 10000)])
    cdf = np.linspace(0, 1, len(sorted_vals))
    axes[2].plot(sorted_vals, cdf, color="navy", lw=1.8)
    axes[2].set_title("Empirical CDF")
    axes[2].set_xlabel("Voxel Intensity")
    axes[2].set_ylabel("Cumulative Fraction")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


def brain_mask(mri, min_intensity="auto"):
    """Otsu threshold on voxels above background floor, holes filled, largest component kept.

    min_intensity: "auto" dynamically determines background cutoff above intercept/floor.
    """
    d = mri.data
    finite = d[np.isfinite(d)]
    if min_intensity == "auto":
        dmin = float(finite.min())
        near_min_count = (finite <= dmin + 1e-4).sum()
        if near_min_count / finite.size > 0.10:
            cutoff = dmin + 0.5
        else:
            cutoff = 0.5
    else:
        cutoff = float(min_intensity)

    fg_candidates = finite[finite > cutoff]
    if len(fg_candidates) == 0:
        raise ValueError(f"brain_mask found no voxels above intensity cutoff {cutoff}")

    t = threshold_otsu(fg_candidates)
    fg = d > t
    fg = ndi.binary_fill_holes(fg)
    lab, n = ndi.label(fg)
    if n == 0:
        raise ValueError("brain_mask found no foreground components")
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def mask_stats(mask, mri):
    """Calculate volumetric, spatial, and geometric properties of a binary mask."""
    vox_mm3 = float(np.prod(mri.voxel_mm))
    n_vox = int(mask.sum())
    vol_mL = round(n_vox * vox_mm3 / 1000.0, 2)
    grid_frac = n_vox / mask.size

    if n_vox == 0:
        return {
            "voxels": 0, "volume_mL": 0.0, "grid_fraction": 0.0,
            "centroid_RAS_mm": None, "bbox_voxels": None, "bbox_RAS_mm": None
        }

    com = np.array(ndi.center_of_mass(mask))
    centroid_ras = tuple(round(float(v), 1) for v in (mri.affine @ np.append(com, 1.0))[:3])

    idx = np.where(mask)
    bbox_vox = {
        "i": (int(idx[0].min()), int(idx[0].max())),
        "j": (int(idx[1].min()), int(idx[1].max())),
        "k": (int(idx[2].min()), int(idx[2].max())),
    }
    corners = np.array([
        [bbox_vox["i"][0], bbox_vox["j"][0], bbox_vox["k"][0], 1],
        [bbox_vox["i"][1], bbox_vox["j"][1], bbox_vox["k"][1], 1],
    ], dtype=float)
    ras_corners = (mri.affine @ corners.T).T[:, :3]
    bbox_ras = {
        "x": (round(float(min(ras_corners[:, 0])), 1), round(float(max(ras_corners[:, 0])), 1)),
        "y": (round(float(min(ras_corners[:, 1])), 1), round(float(max(ras_corners[:, 1])), 1)),
        "z": (round(float(min(ras_corners[:, 2])), 1), round(float(max(ras_corners[:, 2])), 1)),
    }

    return {
        "voxels": n_vox,
        "volume_mL": vol_mL,
        "grid_fraction": round(grid_frac, 4),
        "centroid_RAS_mm": centroid_ras,
        "bbox_voxels": bbox_vox,
        "bbox_RAS_mm": bbox_ras,
    }


def qc_summary(mri_list):
    """Generate a consolidated researcher-facing QC DataFrame across multiple MRI volumes."""
    rows = []
    for m in mri_list:
        q = intensity_qc(m)
        rows.append({
            "file": m.path.name,
            "modality": m.modality,
            "raw_dtype": m.raw_dtype,
            "shape": str(m.shape),
            "voxel_mm": str(tuple(round(v, 2) for v in m.voxel_mm)),
            "orientation": m.orientation,
            "fov_mm": str(tuple(round(v, 1) for v in m.fov_mm)),
            "min": round(q["min"], 3),
            "median": round(q["median"], 1),
            "max": round(q["max"], 1),
            "near_min%": f"{q['near_min_pct']:.1%}",
            "bg_trap": q["bg_trap_flag"],
            "clipping": q["clipping_flag"],
        })
    return pd.DataFrame(rows)


def _direction(mri, direction):
    if direction == "auto":
        return "below" if mri.modality == "ADC" else "above"
    if direction not in ("above", "below"):
        raise ValueError("direction must be 'auto', 'above' or 'below'")
    return direction


def lesion_mask(mri, brain, threshold, direction="auto", min_voxels=10, verbose=True):
    """Voxels inside the brain mask past an explicit threshold, small components removed.

    threshold is required on purpose: lesion volume is very sensitive to it (see threshold_sensitivity).
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number")
    rule = _direction(mri, direction)
    hit = (mri.data < threshold) if rule == "below" else (mri.data > threshold)
    hit &= brain
    lab, n = ndi.label(hit)
    if n == 0:
        if verbose:
            print(f"lesion rule: {rule} {threshold} -> 0 voxels")
        return np.zeros(mri.shape, dtype=bool)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    out = keep[lab]
    if verbose:
        print(f"lesion rule: {rule} {threshold} ({n} raw components, {int(keep.sum())} kept >= {min_voxels} voxels)")
    return out


def lesion_stats(mask, mri):
    """Volume and location of a lesion mask, with one row per connected component."""
    voxel_mm3 = float(np.prod(mri.voxel_mm))
    lab, n = ndi.label(mask)
    summary = {"total_mL": round(float(mask.sum()) * voxel_mm3 / 1000, 2), "n_components": int(n)}
    if n == 0:
        summary["components"] = pd.DataFrame(columns=["component", "voxels", "mL", "x_RAS", "y_RAS", "z_RAS"])
        return summary
    idx = np.arange(1, n + 1)
    sizes = ndi.sum(mask, lab, idx).astype(int)
    coms = np.array(ndi.center_of_mass(mask, lab, idx))
    ras = (mri.affine @ np.column_stack([coms, np.ones(n)]).T).T[:, :3]
    comp = pd.DataFrame({
        "component": idx,
        "voxels": sizes,
        "mL": np.round(sizes * voxel_mm3 / 1000, 2),
        "x_RAS": np.round(ras[:, 0], 1),
        "y_RAS": np.round(ras[:, 1], 1),
        "z_RAS": np.round(ras[:, 2], 1),
    }).sort_values("voxels", ascending=False, ignore_index=True)
    whole = np.array(ndi.center_of_mass(mask))
    centre = (mri.affine @ np.append(whole, 1.0))[:3]
    summary.update({
        "largest_mL": float(comp["mL"].iloc[0]),
        "centroid_RAS_mm": tuple(round(float(v), 1) for v in centre),
        "side (by centroid x_RAS)": "left" if centre[0] < 0 else "right",
        "components": comp,
    })
    return summary


def threshold_sensitivity(mri, brain, thresholds, direction="auto", min_voxels=10):
    """Lesion volume and centroid at each threshold, to see how much the choice matters."""
    rows = []
    for t in thresholds:
        m = lesion_mask(mri, brain, t, direction=direction, min_voxels=min_voxels, verbose=False)
        s = lesion_stats(m, mri)
        rows.append({
            "threshold": t,
            "direction": _direction(mri, direction),
            "lesion_mL": s["total_mL"],
            "n_components": s["n_components"],
            "centroid_RAS_mm": s.get("centroid_RAS_mm"),
        })
    return pd.DataFrame(rows)


def _plane_slice(arr, world_axis, index, affine):
    """2D view of arr perpendicular to world_axis, rows = vertical world axis, columns = horizontal.
    Rows increase with world z (or y for the sagittal plane). Patient right is on the left for x.
    Returns (image, row_array_axis, col_array_axis) so the caller can set the aspect from voxel sizes."""
    A = affine[:3, :3]
    arr_axis = int(np.argmax(np.abs(A[world_axis])))              # array axis running along this world axis
    sl = np.take(arr, index, axis=arr_axis)                       # 2D, remaining array axes in order
    others = [a for a in range(3) if a != arr_axis]
    h_world, v_world = [w for w in range(3) if w != world_axis]   # horizontal, vertical world axes
    a_h = max(others, key=lambda a: abs(A[h_world, a]))           # array axis that runs horizontally
    a_v = others[1] if a_h == others[0] else others[0]
    img = sl if a_v == others[0] else sl.T                        # img[row=a_v, col=a_h]
    if A[v_world, a_v] < 0:                                       # make world z increase upward
        img = img[::-1, :]
    if A[h_world, a_h] < 0:                                       # make world coordinate increase rightward
        img = img[:, ::-1]
    if h_world == 0:                                              # x: patient right on screen left
        img = img[:, ::-1]
    return img, a_v, a_h


def show_mri(mri, mask=None, planes=("axial", "coronal", "sagittal"), index=None):
    """Show one middle slice per plane, with an optional mask outline overlay.

    Slices default to the one with the most mask voxels (or the middle if no mask).
    Orientation comes from the affine: rows go superior at the top, columns go patient right on the left.
    """
    fig, axes = plt.subplots(1, len(planes), figsize=(4.2 * len(planes), 4.4))  # one panel per plane
    axes = np.atleast_1d(axes)
    A = mri.affine[:3, :3]
    for ax, plane in zip(axes, planes):
        w = _PLANE_WORLD_AXIS[plane]
        arr_axis = int(np.argmax(np.abs(A[w])))
        if index is not None:
            i = int(index)
        elif mask is not None and mask.any():
            counts = mask.sum(axis=tuple(a for a in range(3) if a != arr_axis))
            i = int(np.argmax(counts))
        else:
            i = mri.shape[arr_axis] // 2
        img, a_v, a_h = _plane_slice(mri.data, w, i, mri.affine)
        zooms = mri.voxel_mm
        ax.imshow(img, cmap="gray", origin="lower", aspect=zooms[a_v] / zooms[a_h],
                  vmin=np.percentile(mri.data[mri.data > 0.5], 1),
                  vmax=np.percentile(mri.data[mri.data > 0.5], 99.5), interpolation="nearest")
        if mask is not None:
            m = _plane_slice(mask.astype(np.uint8), w, i, mri.affine)[0].astype(bool)
            ax.contour(m, levels=[0.5], colors="red", linewidths=1.2)
        ijk = np.array([s / 2 for s in mri.shape], dtype=float)
        ijk[arr_axis] = i
        mm = (mri.affine @ np.append(ijk, 1.0))[w]
        ax.set_title(f"{plane}  {'xyz'[w]} = {mm:.1f} mm")
        ax.axis("off")
    fig.suptitle(f"{mri.path.name}" + (f"  |  mask: {int(mask.sum()):,} voxels" if mask is not None else ""))
    fig.tight_layout()
    return fig


# =============================================================================
# Section 2: Dual-Modality Diffusion Physics (DWI ↔ ADC)
# =============================================================================

def check_spatial_alignment(mri_a, mri_b, atol=1e-3):
    """Validate spatial grid and affine compatibility between two MRI volumes.

    Accepts MRI objects, nibabel images, or file paths.
    Returns a dict with:
      - aligned (bool): True if shapes, voxel zooms, and affine match within atol.
      - shape_match (bool)
      - voxel_match (bool)
      - affine_match (bool)
      - max_affine_diff (float): max absolute difference across 4x4 affine matrices.
      - origin_diff_mm (tuple): difference in world origin (x, y, z) in mm.
      - message (str): researcher-readable diagnostic summary.
    """
    if isinstance(mri_a, (str, Path)):
        mri_a = load_mri(mri_a)
    if isinstance(mri_b, (str, Path)):
        mri_b = load_mri(mri_b)

    shape_a = mri_a.shape
    shape_b = mri_b.shape
    aff_a = mri_a.affine
    aff_b = mri_b.affine
    if hasattr(mri_a, "voxel_mm"):
        zoom_a = mri_a.voxel_mm
    elif hasattr(mri_a, "header"):
        zoom_a = tuple(float(v) for v in mri_a.header.get_zooms()[:3])
    else:
        zoom_a = tuple(float(v) for v in mri_a.img.header.get_zooms()[:3])

    if hasattr(mri_b, "voxel_mm"):
        zoom_b = mri_b.voxel_mm
    elif hasattr(mri_b, "header"):
        zoom_b = tuple(float(v) for v in mri_b.header.get_zooms()[:3])
    else:
        zoom_b = tuple(float(v) for v in mri_b.img.header.get_zooms()[:3])

    shape_match = shape_a == shape_b
    voxel_match = np.allclose(zoom_a, zoom_b, atol=atol)
    affine_match = np.allclose(aff_a, aff_b, atol=atol)
    max_aff_diff = float(np.max(np.abs(aff_a - aff_b)))
    origin_diff = tuple(float(d) for d in np.abs(aff_a[:3, 3] - aff_b[:3, 3]))

    aligned = shape_match and affine_match

    name_a = getattr(mri_a, "path", None)
    name_a = name_a.name if name_a else "vol_a"
    name_b = getattr(mri_b, "path", None)
    name_b = name_b.name if name_b else "vol_b"

    if aligned:
        msg = f"Aligned: {name_a} and {name_b} share identical grid {shape_a} and affine (max |diff| = {max_aff_diff:.4f} mm)."
    else:
        reasons = []
        if not shape_match:
            reasons.append(f"shape mismatch {shape_a} vs {shape_b}")
        if not voxel_match:
            reasons.append(f"voxel size mismatch {zoom_a} vs {zoom_b}")
        if not affine_match:
            reasons.append(f"affine mismatch (max |diff| = {max_aff_diff:.4f} mm, origin diff = {origin_diff} mm)")
        msg = f"Not aligned: {'; '.join(reasons)}."

    return {
        "aligned": aligned,
        "shape_match": shape_match,
        "voxel_match": voxel_match,
        "affine_match": affine_match,
        "max_affine_diff": max_aff_diff,
        "origin_diff_mm": origin_diff,
        "message": msg,
    }


@dataclass
class DiffusionPair:
    """A co-registered pair of DWI and ADC volumes on an identical spatial grid."""
    dwi: MRI
    adc: MRI
    brain_mask: np.ndarray = None
    name: str = ""

    @property
    def shape(self):
        return self.dwi.shape

    @property
    def affine(self):
        return self.dwi.affine

    @property
    def voxel_mm(self):
        return self.dwi.voxel_mm

    @property
    def voxel_volume_mL(self):
        return np.prod(self.voxel_mm) / 1000.0


def load_diffusion_pair(dwi, adc, compute_brain_mask=True, name=None):
    """Load and validate a co-registered DWI ↔ ADC diffusion pair.

    Accepts file paths or already-loaded MRI objects.
    Computes brain mask from the DWI volume (the optimal anatomical anchor for parenchyma).
    Raises ValueError if volumes are not on an identical grid.
    """
    if not isinstance(dwi, MRI):
        dwi = load_mri(dwi)
    if not isinstance(adc, MRI):
        adc = load_mri(adc)

    align = check_spatial_alignment(dwi, adc)
    if not align["aligned"]:
        raise ValueError(f"DWI and ADC are not spatially aligned: {align['message']}")

    b_mask = brain_mask(dwi) if compute_brain_mask else None
    pair_name = name or f"{dwi.path.name} ↔ {adc.path.name}"
    return DiffusionPair(dwi=dwi, adc=adc, brain_mask=b_mask, name=pair_name)


def paired_diffusion_stats(pair_or_dwi, adc=None, mask=None, regions=None):
    """Extract joint intensity and correlation statistics between DWI and ADC.

    Parameters:
      pair_or_dwi: DiffusionPair or DWI MRI object.
      adc: ADC MRI object if pair_or_dwi is a DWI MRI.
      mask: Optional boolean 3D array (defaults to pair.brain_mask or whole volume).
      regions: Optional dict mapping {region_name: boolean_mask} to analyze multiple ROIs.

    Returns:
      pd.DataFrame if regions is provided, or a dict if single mask analyzed.
    """
    if isinstance(pair_or_dwi, DiffusionPair):
        dwi = pair_or_dwi.dwi
        adc = pair_or_dwi.adc
        default_mask = pair_or_dwi.brain_mask if pair_or_dwi.brain_mask is not None else np.ones(dwi.shape, dtype=bool)
    else:
        dwi = pair_or_dwi
        if adc is None:
            raise ValueError("Must provide adc when pair_or_dwi is not a DiffusionPair.")
        default_mask = np.ones(dwi.shape, dtype=bool)

    vx_vol_mL = np.prod(dwi.voxel_mm) / 1000.0

    def _calc_roi_stats(roi_name, m):
        if m is None or not np.any(m):
            return {
                "region": roi_name, "voxels": 0, "volume_mL": 0.0,
                "dwi_mean": np.nan, "dwi_std": np.nan, "dwi_median": np.nan, "dwi_iqr": np.nan,
                "adc_mean": np.nan, "adc_std": np.nan, "adc_median": np.nan, "adc_iqr": np.nan,
                "pearson_r": np.nan, "spearman_rho": np.nan,
            }
        d_vals = dwi.data[m].astype(float)
        a_vals = adc.data[m].astype(float)
        n_vox = int(np.sum(m))
        vol_mL = round(n_vox * vx_vol_mL, 2)

        if n_vox > 2 and np.std(d_vals) > 1e-6 and np.std(a_vals) > 1e-6:
            r, _ = stats.pearsonr(d_vals, a_vals)
            rho, _ = stats.spearmanr(d_vals, a_vals)
        else:
            r, rho = np.nan, np.nan

        return {
            "region": roi_name,
            "voxels": n_vox,
            "volume_mL": vol_mL,
            "dwi_mean": round(float(np.mean(d_vals)), 1),
            "dwi_std": round(float(np.std(d_vals)), 1),
            "dwi_median": round(float(np.median(d_vals)), 1),
            "dwi_iqr": round(float(np.percentile(d_vals, 75) - np.percentile(d_vals, 25)), 1),
            "adc_mean": round(float(np.mean(a_vals)), 1),
            "adc_std": round(float(np.std(a_vals)), 1),
            "adc_median": round(float(np.median(a_vals)), 1),
            "adc_iqr": round(float(np.percentile(a_vals, 75) - np.percentile(a_vals, 25)), 1),
            "pearson_r": round(float(r), 3) if not np.isnan(r) else np.nan,
            "spearman_rho": round(float(rho), 3) if not np.isnan(rho) else np.nan,
        }

    if regions is not None:
        rows = [_calc_roi_stats(name, r_mask) for name, r_mask in regions.items()]
        return pd.DataFrame(rows)

    m = mask if mask is not None else default_mask
    return _calc_roi_stats("ROI", m)


def restricted_diffusion_lesion(pair_or_dwi, adc=None, brain_mask=None,
                                dwi_threshold=150.0, adc_max=750.0,
                                min_voxels=10, return_all=False):
    """Delineate acute ischemic core candidates using paired DWI hyperintensity and ADC restriction.

    Parameters:
      pair_or_dwi: DiffusionPair or DWI MRI object.
      adc: ADC MRI object if pair_or_dwi is a DWI MRI.
      brain_mask: Boolean brain parenchyma mask (defaults to pair.brain_mask or computed).
      dwi_threshold: Minimum DWI intensity for hyperintensity candidate.
      adc_max: Maximum ADC value for restricted diffusion (cytotoxic edema cutoff).
      min_voxels: Minimum connected cluster size to retain.
      return_all: If True, returns dict with {"core", "shine_through", "dwi_candidate"}.
                  If False, returns boolean core mask.
    """
    if isinstance(pair_or_dwi, DiffusionPair):
        dwi = pair_or_dwi.dwi
        adc = pair_or_dwi.adc
        b_mask = brain_mask if brain_mask is not None else pair_or_dwi.brain_mask
    else:
        dwi = pair_or_dwi
        if adc is None:
            raise ValueError("Must provide adc when pair_or_dwi is not a DiffusionPair.")
        b_mask = brain_mask

    if b_mask is None:
        b_mask = np.ones(dwi.shape, dtype=bool)

    dwi_cand = (dwi.data >= dwi_threshold) & b_mask
    core_raw = dwi_cand & (adc.data <= adc_max)
    shine_raw = dwi_cand & (adc.data > adc_max)

    def _filter_min_voxels(raw_mask):
        if not np.any(raw_mask) or min_voxels <= 1:
            return raw_mask
        labeled, n_comp = ndi.label(raw_mask)
        if n_comp == 0:
            return np.zeros_like(raw_mask, dtype=bool)
        sizes = ndi.sum(raw_mask, labeled, range(1, n_comp + 1))
        keep_labels = np.where(np.asarray(sizes) >= min_voxels)[0] + 1
        return np.isin(labeled, keep_labels)

    core_filt = _filter_min_voxels(core_raw)
    shine_filt = _filter_min_voxels(shine_raw)
    dwi_filt = _filter_min_voxels(dwi_cand)

    if return_all:
        return {
            "core": core_filt,
            "shine_through": shine_filt,
            "dwi_candidate": dwi_filt,
        }
    return core_filt


def paired_threshold_sensitivity(pair_or_dwi, adc=None, brain_mask=None,
                                 dwi_thresholds=(100, 150, 200, 250),
                                 adc_cutoffs=(650, 700, 750, 800),
                                 min_voxels=10):
    """Explore the 2D threshold sensitivity landscape across DWI thresholds and ADC cutoffs.

    Returns a pd.DataFrame with core volume, shine-through volume, core retention %, and centroids.
    """
    if isinstance(pair_or_dwi, DiffusionPair):
        dwi = pair_or_dwi.dwi
        adc = pair_or_dwi.adc
        b_mask = brain_mask if brain_mask is not None else pair_or_dwi.brain_mask
    else:
        dwi = pair_or_dwi
        if adc is None:
            raise ValueError("Must provide adc when pair_or_dwi is not a DiffusionPair.")
        b_mask = brain_mask

    if b_mask is None:
        b_mask = np.ones(dwi.shape, dtype=bool)

    rows = []
    for dt in dwi_thresholds:
        for at in adc_cutoffs:
            masks = restricted_diffusion_lesion(dwi, adc=adc, brain_mask=b_mask,
                                               dwi_threshold=dt, adc_max=at,
                                               min_voxels=min_voxels, return_all=True)
            core_m = masks["core"]
            dwi_m = masks["dwi_candidate"]
            shine_m = masks["shine_through"]

            core_s = lesion_stats(core_m, dwi)
            dwi_s = lesion_stats(dwi_m, dwi)
            shine_s = lesion_stats(shine_m, dwi)

            dwi_vol = dwi_s["total_mL"]
            core_vol = core_s["total_mL"]
            shine_vol = shine_s["total_mL"]
            retention = round((core_vol / dwi_vol * 100.0), 1) if dwi_vol > 0 else 0.0

            rows.append({
                "dwi_threshold": dt,
                "adc_cutoff": at,
                "dwi_alone_mL": dwi_vol,
                "core_mL": core_vol,
                "shine_through_mL": shine_vol,
                "core_retention_pct": retention,
                "n_core_components": core_s["n_components"],
                "centroid_RAS_mm": core_s.get("centroid_RAS_mm"),
            })
    return pd.DataFrame(rows)


def plot_joint_diffusion(pair_or_dwi, adc=None, mask=None,
                         dwi_threshold=None, adc_cutoff=None,
                         candidate_mask=None, bins=80, figsize=(13, 5), title=None):
    """Visualize 2D joint DWI vs ADC density and comparative marginal distributions.

    Panel 1: 2D joint density (hist2d log-scaled) in brain parenchyma, with quadrant annotations.
    Panel 2: Comparative ADC distributions (parenchyma vs candidate/core).
    """
    if isinstance(pair_or_dwi, DiffusionPair):
        dwi = pair_or_dwi.dwi
        adc = pair_or_dwi.adc
        parenchyma = mask if mask is not None else pair_or_dwi.brain_mask
        name = pair_or_dwi.name
    else:
        dwi = pair_or_dwi
        if adc is None:
            raise ValueError("Must provide adc when pair_or_dwi is not a DiffusionPair.")
        parenchyma = mask
        name = f"{dwi.path.name} ↔ {adc.path.name}"

    if parenchyma is None:
        parenchyma = np.ones(dwi.shape, dtype=bool)

    d_par = dwi.data[parenchyma].ravel()
    a_par = adc.data[parenchyma].ravel()

    valid = np.isfinite(d_par) & np.isfinite(a_par) & (d_par > 0) & (a_par > 0)
    d_par = d_par[valid]
    a_par = a_par[valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    d_max = float(np.percentile(d_par, 99.8))
    a_max = float(np.percentile(a_par, 99.8))
    h = ax1.hist2d(d_par, a_par, bins=bins, range=[[0, d_max * 1.1], [0, a_max * 1.1]],
                   cmap="inferno", norm=LogNorm())
    fig.colorbar(h[3], ax=ax1, label="Voxel count (log scale)")
    ax1.set_xlabel("DWI Intensity (a.u.)", fontsize=11)
    ax1.set_ylabel("ADC Value ($10^{-6}\\ mm^2/s$)", fontsize=11)
    ax1.set_title("Joint DWI–ADC Parenchymal Distribution", fontsize=12)

    if candidate_mask is not None and np.any(candidate_mask):
        d_cand = dwi.data[candidate_mask].ravel()
        a_cand = adc.data[candidate_mask].ravel()
        if len(d_cand) > 5000:
            idx = np.random.choice(len(d_cand), 5000, replace=False)
            d_cand, a_cand = d_cand[idx], a_cand[idx]
        ax1.scatter(d_cand, a_cand, s=3, color="cyan", alpha=0.4, label="Candidate Core")
        ax1.legend(loc="upper left")

    if dwi_threshold is not None:
        ax1.axvline(dwi_threshold, color="white", linestyle="--", linewidth=1.2,
                    label=f"DWI > {dwi_threshold}")
    if adc_cutoff is not None:
        ax1.axhline(adc_cutoff, color="cyan", linestyle="--", linewidth=1.2,
                    label=f"ADC < {adc_cutoff}")

    if dwi_threshold is not None and adc_cutoff is not None:
        ax1.text(dwi_threshold + 5, adc_cutoff - 50, "Cytotoxic Core\n(DWI high, ADC low)",
                 color="white", fontsize=8, fontweight="bold", verticalalignment="top")
        ax1.text(dwi_threshold + 5, adc_cutoff + 50, "T2 Shine-Through\n(DWI high, ADC high)",
                 color="white", fontsize=8, fontweight="bold", verticalalignment="bottom")

    ax2.hist(a_par, bins=60, range=[0, a_max * 1.1], density=True, alpha=0.5,
             color="gray", label=f"Brain Parenchyma (med={np.median(a_par):.0f})")

    if dwi_threshold is not None:
        dwi_high = parenchyma & (dwi.data >= dwi_threshold)
        if np.any(dwi_high):
            ax2.hist(adc.data[dwi_high].ravel(), bins=60, range=[0, a_max * 1.1], density=True,
                     histtype="step", linewidth=1.8, color="red",
                     label=f"DWI ≥ {dwi_threshold} (med={np.median(adc.data[dwi_high]):.0f})")

    if candidate_mask is not None and np.any(candidate_mask):
        ax2.hist(adc.data[candidate_mask].ravel(), bins=60, range=[0, a_max * 1.1], density=True,
                 histtype="step", linewidth=2.0, color="cyan",
                 label=f"Core Mask (med={np.median(adc.data[candidate_mask]):.0f})")

    if adc_cutoff is not None:
        ax2.axvline(adc_cutoff, color="cyan", linestyle="--", linewidth=1.2,
                    label=f"ADC Cutoff ({adc_cutoff})")

    ax2.set_xlabel("ADC Value ($10^{-6}\\ mm^2/s$)", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.set_title("ADC Distribution Shift in Stroke Region", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle(title or f"Dual-Modality Diffusion Physics: {name}", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def show_diffusion_pair(pair_or_dwi, adc=None, core_mask=None, shine_mask=None,
                        planes=("axial", "coronal", "sagittal"), index=None):
    """Side-by-side multiplanar display of DWI and ADC with core and shine-through overlays.

    - DWI shown on top row, ADC shown on bottom row.
    - core_mask outlined in red.
    - shine_mask outlined in gold/yellow.
    - Matches orientation conventions of show_mri.
    """
    if isinstance(pair_or_dwi, DiffusionPair):
        dwi = pair_or_dwi.dwi
        adc = pair_or_dwi.adc
    else:
        dwi = pair_or_dwi
        if adc is None:
            raise ValueError("Must provide adc when pair_or_dwi is not a DiffusionPair.")

    fig, axes = plt.subplots(2, len(planes), figsize=(4.2 * len(planes), 8.5))
    axes = np.atleast_2d(axes)
    A = dwi.affine[:3, :3]

    ref_mask = core_mask if (core_mask is not None and core_mask.any()) else shine_mask

    for col, plane in enumerate(planes):
        w = _PLANE_WORLD_AXIS[plane]
        arr_axis = int(np.argmax(np.abs(A[w])))

        if index is not None:
            i = int(index)
        elif ref_mask is not None and ref_mask.any():
            counts = ref_mask.sum(axis=tuple(a for a in range(3) if a != arr_axis))
            i = int(np.argmax(counts))
        else:
            i = dwi.shape[arr_axis] // 2

        img_dwi, a_v, a_h = _plane_slice(dwi.data, w, i, dwi.affine)
        img_adc, _, _ = _plane_slice(adc.data, w, i, adc.affine)
        zooms = dwi.voxel_mm
        aspect = zooms[a_v] / zooms[a_h]

        ijk = np.array([s / 2 for s in dwi.shape], dtype=float)
        ijk[arr_axis] = i
        mm = (dwi.affine @ np.append(ijk, 1.0))[w]

        ax_d = axes[0, col]
        ax_d.imshow(img_dwi, cmap="gray", origin="lower", aspect=aspect,
                    vmin=np.percentile(dwi.data[dwi.data > 0.5], 1),
                    vmax=np.percentile(dwi.data[dwi.data > 0.5], 99.5), interpolation="nearest")
        ax_d.set_title(f"DWI | {plane}  {'xyz'[w]} = {mm:.1f} mm")
        ax_d.axis("off")

        ax_a = axes[1, col]
        ax_a.imshow(img_adc, cmap="gray", origin="lower", aspect=aspect,
                    vmin=np.percentile(adc.data[adc.data > 0.5], 1),
                    vmax=np.percentile(adc.data[adc.data > 0.5], 99.5), interpolation="nearest")
        ax_a.set_title(f"ADC | {plane}  {'xyz'[w]} = {mm:.1f} mm")
        ax_a.axis("off")

        for ax in (ax_d, ax_a):
            if core_mask is not None and core_mask.any():
                m_core = _plane_slice(core_mask.astype(np.uint8), w, i, dwi.affine)[0].astype(bool)
                if m_core.any():
                    ax.contour(m_core, levels=[0.5], colors="red", linewidths=1.3)
            if shine_mask is not None and shine_mask.any():
                m_shine = _plane_slice(shine_mask.astype(np.uint8), w, i, dwi.affine)[0].astype(bool)
                if m_shine.any():
                    ax.contour(m_shine, levels=[0.5], colors="gold", linewidths=1.2)

    legend_elements = []
    if core_mask is not None and core_mask.any():
        legend_elements.append("Red: Restricted Core")
    if shine_mask is not None and shine_mask.any():
        legend_elements.append("Gold: T2 Shine-Through")
    subtitle = f"  ({', '.join(legend_elements)})" if legend_elements else ""
    fig.suptitle(f"Diffusion Physics Co-Registration: {dwi.path.name} & {adc.path.name}{subtitle}", fontsize=13)
    fig.tight_layout()
    return fig


def compare_mri_volumes(mri_a, mri_b, mask_a=None, mask_b=None, label_a="A", label_b="B"):
    """Voxelwise and volumetric comparison between two co-registered MRI volumes.

    Use this for longitudinal comparisons (e.g. 16 dpm vs 11 dpm), for comparing two acquisitions at one
    timepoint, or for comparing two lesion delineations - anything that needs two volumes confirmed to share
    a grid. Raises ValueError if they are not aligned (see check_spatial_alignment): comparing volumes that
    are not truly on the same grid would silently fabricate a spatial relationship.

    mask_a, mask_b  optional boolean masks (e.g. from lesion_mask() at each timepoint). If both are given,
                   also reports Dice/Jaccard overlap and the volume change between them.

    Returns a dict: difference (voxelwise mri_b - mri_a array), volume_mL_<label> for each mask given, and
    (when both masks are given) dice, jaccard, volume_change_mL.
    """
    align = check_spatial_alignment(mri_a, mri_b)
    if not align["aligned"]:
        raise ValueError(f"{label_a} and {label_b} are not spatially aligned: {align['message']}")

    result = {"label_a": label_a, "label_b": label_b, "difference": mri_b.data - mri_a.data}
    voxel_mm3 = float(np.prod(mri_a.voxel_mm))

    if mask_a is not None:
        mask_a = np.asarray(mask_a, dtype=bool)
        result[f"mean_intensity_{label_a}_in_mask_a"] = float(np.mean(mri_a.data[mask_a]))
        result[f"mean_intensity_{label_b}_in_mask_a"] = float(np.mean(mri_b.data[mask_a]))
        result[f"volume_mL_{label_a}"] = round(float(mask_a.sum()) * voxel_mm3 / 1000, 2)
    if mask_b is not None:
        mask_b = np.asarray(mask_b, dtype=bool)
        result[f"volume_mL_{label_b}"] = round(float(mask_b.sum()) * voxel_mm3 / 1000, 2)
    if mask_a is not None and mask_b is not None:
        inter = int(np.logical_and(mask_a, mask_b).sum())
        union = int(np.logical_or(mask_a, mask_b).sum())
        denom = int(mask_a.sum() + mask_b.sum())
        result["dice"] = round(2 * inter / denom, 3) if denom > 0 else float("nan")
        result["jaccard"] = round(inter / union, 3) if union > 0 else float("nan")
        result["volume_change_mL"] = round((int(mask_b.sum()) - int(mask_a.sum())) * voxel_mm3 / 1000, 2)
    return result


def lesion_border_profile(mri, lesion_mask, distance_range_mm=(-10, 10), step_mm=1.0):
    """Mean/median MRI intensity as a function of signed distance from a lesion mask's boundary.

    distance_range_mm is (inside, outside): negative values are inside the lesion, positive values are
    outside it in normal-appearing tissue, so the returned profile reads left-to-right as core -> edge -> rim.
    Uses scipy.ndimage.distance_transform_edt with the real voxel spacing, so the distance axis is in mm
    regardless of the volume's voxel size.

    lesion_mask must be on mri's own grid (mri.shape) - this is a single-modality MRI operation; it does not
    cross into histology/annotation space (see manifest.json's krow limitation, documented at
    map_voxel_to_histology_section, for why that cross-modality version isn't offered as a precise utility).

    Returns a DataFrame: distance_mm (bin centre), n_voxels, mean_intensity, median_intensity.
    """
    mask = np.asarray(lesion_mask, dtype=bool)
    if mask.shape != mri.shape:
        raise ValueError(f"lesion_mask shape {mask.shape} must match mri.shape {mri.shape}")

    sampling = mri.voxel_mm
    dist_outside = ndi.distance_transform_edt(~mask, sampling=sampling)
    dist_inside = ndi.distance_transform_edt(mask, sampling=sampling)
    signed = dist_outside - dist_inside   # negative inside the mask, positive outside

    lo, hi = distance_range_mm
    edges = np.arange(lo, hi + step_mm, step_mm)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (signed >= a) & (signed < b)
        vals = mri.data[sel]
        vals = vals[np.isfinite(vals)]
        rows.append({
            "distance_mm": round(float((a + b) / 2), 2),
            "n_voxels": int(sel.sum()),
            "mean_intensity": float(np.mean(vals)) if len(vals) else np.nan,
            "median_intensity": float(np.median(vals)) if len(vals) else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Section 3: MRI <-> Histology / Atlas Correspondence (manifest.json)
# =============================================================================
#
# manifest.json (built for the frontend viewer) carries the block-affine geometry that bridges MRI/reference
# voxels to physical histology blocks, sections and global section IDs, plus pre-aligned MNI/arterial label
# volumes. Pass in the already-loaded dict (json.load(open("manifest.json"))); nothing below reads a path
# that isn't given to it explicitly.
#
# Caveat that shapes what is and isn't offered here: manifest["blocks"][...]["krow"] only resolves the
# OUT-OF-PLANE coordinate (which oblique histology slice a reference point falls on). It does not give the
# full in-plane 2D placement needed to warp a histology section's pixel grid precisely onto the MRI volume -
# there is no evidence of that registration in manifest.json or sets_data.json. That is why these functions
# stop at "which section" rather than "which exact MRI voxel underlies this annotation polygon".

def _align_to_manifest_shape(arr, target_shape):
    """Permute `arr`'s axes to match `target_shape`, by verifying the match rather than assuming an order.

    Reference-space arrays here don't share one array axis convention: a NIfTI volume already resampled onto
    manifest.json's reference grid (e.g. iv11_DWI.nii.gz, shape (377, 211, 238)) stores its voxel axes in a
    different order than manifest["space"]["dims"] (211, 377, 238) / manifest["space"]["axes"] (i, j, k).
    Since the three reference-space dimensions are all different sizes, the permutation can be derived from
    shape alone and verified - never hardcoded, never silently assumed.
    """
    target_shape = tuple(target_shape)
    if arr.shape == target_shape:
        return arr
    if sorted(arr.shape) != sorted(target_shape):
        raise ValueError(f"array shape {arr.shape} is not a permutation of reference space dims {target_shape}")
    perm = tuple(arr.shape.index(s) for s in target_shape)
    if len(set(perm)) != len(target_shape):
        raise ValueError("ambiguous axis permutation (repeated dimension size); pass an array already "
                         "aligned to manifest['space']['dims']")
    return np.transpose(arr, perm)


def _resolve_manifest_points(mask_or_points, manifest):
    """Points in manifest['space']['axes'] (i, j, k) order, from either a 3D mask or an (N, 3) point array."""
    arr = np.asarray(mask_or_points)
    dims = tuple(manifest["space"]["dims"])
    if arr.ndim == 3:
        aligned = _align_to_manifest_shape(arr.astype(bool), dims)
        return np.argwhere(aligned).astype(float)
    pts = np.atleast_2d(np.asarray(mask_or_points, dtype=float))
    if pts.shape[-1] != 3:
        raise ValueError("points_ijk must have shape (3,) or (N, 3), in manifest['space']['axes'] (i, j, k)")
    return pts


_LATERALITY_WORDS = {"left": "right", "right": "left", "Left": "Right", "Right": "Left",
                     "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _swap_laterality(name):
    return re.sub(r"\b(left|right|Left|Right|LEFT|RIGHT)\b", lambda m: _LATERALITY_WORDS[m.group(0)], name)


def load_manifest_volume(manifest, key, data_dir=None):
    """Decode one of manifest.json's flat, gzip-compressed label/segmentation volumes.

    key: "mni" or "art" (manifest["volumes"]["labels"]) or "segs" (manifest["volumes"]["segs"]["file"]).

    Returns a uint8 array shaped manifest["space"]["dims"], indexed vol[i, j, k] to match
    manifest["space"]["axes"] exactly - NOT necessarily the axis order of any particular NIfTI file (see
    _align_to_manifest_shape, used by regional_lesion_burden to reconcile the two).

    These are categorical volumes (atlas label ids, lesion-mask bits) for lookup only. MRI intensity analysis
    should use the real NIfTI files via load_mri - the manifest's "gray" series volumes are lossy uint8
    display windows, not scientific intensity data.
    """
    if key in ("mni", "art"):
        fname = manifest["volumes"]["labels"][key]
    elif key == "segs":
        fname = manifest["volumes"]["segs"]["file"]
    else:
        raise ValueError(f"Unknown manifest volume key {key!r}; expected 'mni', 'art', or 'segs'.")

    if data_dir is not None and (Path(data_dir) / fname).exists():
        path = Path(data_dir) / fname
    else:
        path = _resolve_data_path(fname)
    with gzip.open(path, "rb") as f:
        raw = f.read()

    ni, nj, nk = manifest["space"]["dims"]
    expected = ni * nj * nk
    if len(raw) != expected:
        raise ValueError(f"{path} decompressed to {len(raw)} bytes, expected {expected} "
                         f"({ni}x{nj}x{nk} uint8 voxels) from manifest['space']['dims'].")
    # Flat layout is k-major, then i, then j (manifest["space"]["axes"]); transpose to (i, j, k).
    vol = np.frombuffer(raw, dtype=np.uint8).reshape(nk, ni, nj)
    return np.transpose(vol, (1, 2, 0)).copy()


def regional_lesion_burden(lesion_mask, label_volume, label_lookup, manifest, lr_swap=False):
    """Volumetric lesion burden across anatomical or arterial territories, via a pre-aligned label volume.

    lesion_mask    boolean array from lesion_mask()/restricted_diffusion_lesion(), on any axis order that is
                  a permutation of manifest["space"]["dims"] (see _align_to_manifest_shape).
    label_volume   from load_manifest_volume(manifest, "mni"/"art"), already in manifest['space']['dims'] order.
    label_lookup   manifest["atlas"]["mni"] or manifest["atlas"]["art"]: {label_id_str: [name, r, g, b, ...]}.
    lr_swap        manifest["notes"]["arterial_lr"] documents that the delivered arterial atlas is mirrored
                  in laterality. Default False reports the raw (mirrored) names; set True to report them
                  left/right-swapped instead. Either way the choice is explicit, never silently applied.

    Returns a DataFrame ranked by lesion_volume_mL: label_id, name, region_volume_mL, lesion_volume_mL,
    pct_region_infiltrated.
    """
    dims = tuple(manifest["space"]["dims"])
    if label_volume.shape != dims:
        raise ValueError(f"label_volume shape {label_volume.shape} must equal manifest['space']['dims'] "
                         f"{dims}; load it with load_manifest_volume(manifest, ...).")
    mask = _align_to_manifest_shape(np.asarray(lesion_mask, dtype=bool), dims)

    voxel_mm3 = float(np.prod(manifest["space"]["spacing"]))
    n_labels = int(label_volume.max()) + 1
    total_counts = np.bincount(label_volume.ravel(), minlength=n_labels)
    lesion_counts = np.bincount(label_volume[mask].ravel(), minlength=n_labels)

    rows = []
    for label_id_str, entry in label_lookup.items():
        label_id = int(label_id_str)
        if label_id == 0 or label_id >= len(total_counts):
            continue
        total_vox = int(total_counts[label_id])
        lesion_vox = int(lesion_counts[label_id]) if label_id < len(lesion_counts) else 0
        if total_vox == 0 and lesion_vox == 0:
            continue
        name = _swap_laterality(entry[0]) if lr_swap else entry[0]
        rows.append({
            "label_id": label_id,
            "name": name,
            "region_volume_mL": round(total_vox * voxel_mm3 / 1000, 2),
            "lesion_volume_mL": round(lesion_vox * voxel_mm3 / 1000, 2),
            "pct_region_infiltrated": round(100 * lesion_vox / total_vox, 2) if total_vox > 0 else 0.0,
        })
    df = pd.DataFrame(rows, columns=["label_id", "name", "region_volume_mL", "lesion_volume_mL",
                                     "pct_region_infiltrated"])
    return df.sort_values("lesion_volume_mL", ascending=False, ignore_index=True)


def map_voxel_to_histology_section(mask_or_points, manifest, block_id=None, section_thickness_mm=None):
    """Map reference-space voxels to their histology block, section number, and global section ID.

    mask_or_points  a 3D boolean mask (e.g. a lesion mask - any axis order that is a permutation of
                   manifest["space"]["dims"], see _align_to_manifest_shape), or an (N, 3) / (3,) array of
                   points already in manifest['space']['axes'] (i, j, k) order.
    block_id        restrict to one biosample block (e.g. 313); default None checks every block in
                   manifest["blocks"] and, per point, picks the one where the point sits most centrally
                   within its valid slice range (largest margin from either edge) - the same rule the
                   viewer uses, so a point is never silently assigned to the wrong block.
    section_thickness_mm  default manifest["section_thickness_mm"] (0.02 mm / 20 um).

    Returns a DataFrame, one row per input point: i, j, k, block_id, bk (oblique in-block slice
    coordinate), mm (physical position along the block's cutting axis), section, global_section_id, margin
    (distance from the nearest edge of the block's valid range - negative means outside it), in_range.
    Points outside every block's range still get a row (with in_range=False), never an exception or a
    silently-wrong "nearest" guess.
    """
    pts = _resolve_manifest_points(mask_or_points, manifest)
    cols = ["i", "j", "k", "block_id", "bk", "mm", "section", "global_section_id", "margin", "in_range"]
    n_pts = pts.shape[0]
    if n_pts == 0:
        return pd.DataFrame(columns=cols)

    thickness = (float(section_thickness_mm) if section_thickness_mm is not None
                else float(manifest["section_thickness_mm"]))
    blocks = manifest["blocks"]
    candidates = {str(block_id): blocks[str(block_id)]} if block_id is not None else blocks
    keys = list(candidates.keys())

    bk = np.empty((n_pts, len(keys)))
    margin = np.empty((n_pts, len(keys)))
    for bi, key in enumerate(keys):
        b = candidates[key]
        a, b_coef, c_coef, d = b["krow"]
        bk[:, bi] = pts[:, 0] * a + pts[:, 1] * b_coef + pts[:, 2] * c_coef + d
        margin[:, bi] = np.minimum(bk[:, bi], b["nslices"] - 1 - bk[:, bi])

    choice = np.argmax(margin, axis=1)
    block_id_out = np.empty(n_pts, dtype=int)
    bk_out = np.empty(n_pts)
    mm_out = np.empty(n_pts)
    section_out = np.empty(n_pts, dtype=int)
    gid_out = np.empty(n_pts, dtype=int)
    margin_out = np.empty(n_pts)

    for bi, key in enumerate(keys):
        sel = choice == bi
        if not np.any(sel):
            continue
        b = candidates[key]
        bk_sel = bk[sel, bi]
        mm_sel = b["mm_slope"] * bk_sel + b["mm_int"]
        sec_dir = 1 if b["mm_at_secmax"] >= b["mm_at_secmin"] else -1
        gid_dir = 1 if b["gid_at_secmax"] >= b["gid_at_secmin"] else -1
        sec_sel = b["sec_min"] + sec_dir * np.round((mm_sel - b["mm_at_secmin"]) / thickness).astype(int)
        sec_sel = np.clip(sec_sel, b["sec_min"], b["sec_max"])
        gid_sel = b["gid_at_secmin"] + gid_dir * (sec_sel - b["sec_min"])

        block_id_out[sel] = int(b["biosample"])
        bk_out[sel] = bk_sel
        mm_out[sel] = mm_sel
        section_out[sel] = sec_sel
        gid_out[sel] = gid_sel
        margin_out[sel] = margin[sel, bi]

    return pd.DataFrame({
        "i": pts[:, 0], "j": pts[:, 1], "k": pts[:, 2],
        "block_id": block_id_out,
        "bk": np.round(bk_out, 2),
        "mm": np.round(mm_out, 3),
        "section": section_out,
        "global_section_id": gid_out,
        "margin": np.round(margin_out, 2),
        "in_range": margin_out >= 0,
    })


def extract_block_plane_from_mri(mri, manifest, block_id, section=None, global_section_id=None, grid_size=None):
    """Resample `mri` along the true oblique cutting plane of one histology section.

    Exactly one of `section` or `global_section_id` must be given, for block `block_id`
    (manifest["blocks"]). Because histology blocks are cut obliquely relative to the reference grid (see the
    module-level caveat above), the nearest cardinal MRI slice is a visibly wrong comparison; this samples
    the real cutting plane via scipy.ndimage.map_coordinates (trilinear), so it can be shown next to the
    actual histology image for that section.

    grid_size  (n_u, n_v) samples along the plane's two in-plane axes; default spans the reference volume's
              own (i, j) extent at 1-voxel steps.

    Returns {"image": 2D array, "spacing_mm": (voxel i, voxel j spacing), "block_id", "section",
    "global_section_id", "bk"}.
    """
    if (section is None) == (global_section_id is None):
        raise ValueError("Pass exactly one of `section` or `global_section_id`.")

    b = manifest["blocks"][str(block_id)]
    thickness = float(manifest["section_thickness_mm"])
    sec_dir = 1 if b["mm_at_secmax"] >= b["mm_at_secmin"] else -1
    gid_dir = 1 if b["gid_at_secmax"] >= b["gid_at_secmin"] else -1

    if global_section_id is not None:
        section = b["sec_min"] + gid_dir * (int(global_section_id) - b["gid_at_secmin"])
    section = int(np.clip(section, b["sec_min"], b["sec_max"]))
    global_section_id = b["gid_at_secmin"] + gid_dir * (section - b["sec_min"])

    mm = b["mm_at_secmin"] + sec_dir * (section - b["sec_min"]) * thickness
    bk_target = (mm - b["mm_int"]) / b["mm_slope"]

    dims = tuple(manifest["space"]["dims"])
    if sorted(mri.shape) != sorted(dims):
        raise ValueError(f"mri.shape {mri.shape} is not a permutation of manifest['space']['dims'] {dims}; "
                         f"this MRI volume isn't on the reference grid these blocks are defined on.")
    spacing = manifest["space"]["spacing"]
    ni, nj, nk = dims
    grid_size = grid_size or (ni, nj)

    a, b_coef, c_coef, d = b["krow"]
    n = np.array([a, b_coef, c_coef], dtype=float)
    n_hat = n / np.linalg.norm(n)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n_hat, ref); u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)

    center = np.array(dims, dtype=float) / 2.0
    t = (bk_target - (n @ center + d)) / (n @ n)
    point0 = center + t * n

    su = np.arange(grid_size[0]) - grid_size[0] / 2.0
    sv = np.arange(grid_size[1]) - grid_size[1] / 2.0
    grid_u, grid_v = np.meshgrid(su, sv, indexing="ij")
    manifest_ijk = point0[None, None, :] + grid_u[..., None] * u + grid_v[..., None] * v

    mri_perm = tuple(mri.shape.index(dim) for dim in dims)  # manifest axis a -> mri array axis mri_perm[a]
    coords = np.stack([manifest_ijk[..., mri_perm.index(ax)] for ax in range(3)], axis=0).reshape(3, -1)

    image = ndi.map_coordinates(mri.data, coords, order=1, mode="constant", cval=0.0)
    image = image.reshape(grid_size)

    return {
        "image": image,
        "spacing_mm": (float(spacing[0]), float(spacing[1])),
        "block_id": int(b["biosample"]),
        "section": section,
        "global_section_id": int(global_section_id),
        "bk": float(bk_target),
    }

