"""roi_utils - small helpers for exploring NumPy ROI stacks and using them in ML.

An ROI stack is a NumPy array of sections stacked along axis 0:
    (Z, H, W)      grayscale
    (Z, H, W, C)   multi-channel / RGB (C = 1, 3 or 4)

Typical use:
    from roi_utils import *

    roi = load_and_preview_roi("<drive file id or url>")   # from Google Drive
    roi = load_roi("my_roi.npy")                            # or from disk

    roi_info(roi)
    preview_sections(roi, start=100, end=150)               # sections 100..149
    preview_sections(roi, sections=[10, 25, 80])

    x = normalize_roi(center_crop(roi, 256))
    t = to_torch(x)                                         # (N, C, H, W) tensor

QC helpers (RGB brightfield stacks, white background):
    st = section_stats(roi)                  # dict of per-section arrays: coverage, bbox, centroid, ...
    flags = flag_sections(st)                # [(section, reason, value, (lo, hi)), ...]
    d = adjacent_change(roi)                 # 1 - NCC between sections N and N+1 (len Z-1)

Requires numpy (scipy for the QC helpers). matplotlib (previews), requests (Google Drive) and torch
(to_torch) are imported only when those functions are used.
"""

import os
import re
import tempfile

import numpy as np

__all__ = [
    "load_roi",
    "load_and_preview_roi",
    "preview_sections",
    "get_sections",
    "roi_info",
    "normalize_roi",
    "crop_roi",
    "center_crop",
    "downsample_roi",
    "to_grayscale",
    "to_torch",
    "to_od",
    "tissue_mask",
    "section_stats",
    "local_zscore",
    "flag_sections",
    "lowres_stack",
    "adjacent_change",
]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _n_sections(roi):
    return roi.shape[0] if roi.ndim >= 3 else 1


def _select(n, start=None, end=None, step=1, sections=None):
    """Return a list of valid section indices (end is exclusive, like range())."""
    if sections is not None:
        idx = [int(s) + n if int(s) < 0 else int(s) for s in np.atleast_1d(sections)]
        good = [i for i in idx if 0 <= i < n]
        if len(good) < len(idx):
            print(f"Warning: ignored sections outside 0..{n - 1}: "
                  f"{sorted(set(idx) - set(good))}")
        if not good:
            raise IndexError(f"No valid sections; this stack has sections 0..{n - 1}.")
        return good
    idx = list(range(*slice(start, end, step).indices(n)))
    if not idx:
        raise IndexError(f"Empty section range; this stack has sections 0..{n - 1}.")
    return idx


def _show(ax, sl):
    if sl.ndim == 3 and sl.shape[-1] == 1:
        ax.imshow(sl[..., 0], cmap="gray")
    elif sl.ndim == 3 and sl.shape[-1] in (3, 4):
        ax.imshow(sl)
    else:
        ax.imshow(sl, cmap="gray")


def _channels_last(roi):
    return roi.ndim == 4


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_roi(path, mmap=False):
    """Load a .npy ROI stack from disk.

    mmap=True memory-maps the file (read-only, nothing is read until used) -
    use it for stacks too large for RAM; slicing then only reads what it needs.
    """
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def _parse_filename(base):
    """Best-effort metadata from an ROI Extractor filename."""
    meta = {}
    m = (re.search(r"(?:biosample|sample|brain)[-_=]?([a-zA-Z0-9]+)", base, re.I)
         or re.search(r"^(Q\d+)", base))
    if m:
        meta["Brain sample / biosample"] = m.group(1)
    m = re.search(r"(?:^|[-_&])[xX][-_=]?(\d+)", base)
    if m:
        meta["Center X"] = int(m.group(1))
    m = re.search(r"(?:^|[-_&])[yY][-_=]?(\d+)", base)
    if m:
        meta["Center Y"] = int(m.group(1))
    z = []
    m = re.search(r"(?:SE|se|sec|section(?:_number)?)[-_=]?(\d+)", base)
    if m:
        z.append(f"Section {m.group(1)}")
    m = re.search(r"(?:^|[-_&])[zZ][-_=]?(\d+)", base)
    if m:
        z.append(f"Z={m.group(1)}")
    if z:
        meta["Z / section information"] = ", ".join(z)
    m = re.search(r"(?:stain[-_=]?([a-zA-Z0-9]+))|(NISL|HEOS|MYEL|IHCS|HE|DAPI)", base, re.I)
    if m:
        meta["Stain"] = (m.group(1) or m.group(2)).upper()
    m = re.search(r"(?:resolution|res)[-_=]?([a-zA-Z0-9]+)|(?:^|[-_&])L(\d+)", base, re.I)
    if m:
        meta["Resolution"] = m.group(1) or f"Level {m.group(2)}"
    return meta


def _download_drive(file_id, dest_path):
    """Stream a public Google Drive file to dest_path; return its filename."""
    import requests

    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    res = session.get(url, stream=True)

    # Large files: Drive returns an HTML "virus scan" page that needs confirming.
    if "text/html" in res.headers.get("Content-Type", ""):
        html = res.content.decode("utf-8", errors="ignore")
        action = re.search(r'action="([^"]+)"', html)
        if action:
            params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html))
            res = session.get(action.group(1), params=params, stream=True)
        else:
            confirm = re.search(r"confirm=([0-9A-Za-z_]+)", html)
            if confirm:
                res = session.get(f"{url}&confirm={confirm.group(1)}", stream=True)
    res.raise_for_status()
    if "text/html" in res.headers.get("Content-Type", ""):
        raise RuntimeError("Google Drive returned a web page instead of the file. "
                           "Check the ID and that sharing is 'Anyone with the link'.")

    filename = f"ROI_{file_id}.npy"
    cd = res.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';\r\n]+)', cd)
    if m:
        filename = m.group(1).strip("'\"")

    with open(dest_path, "wb") as f:  # straight to disk: no in-memory duplicate
        for chunk in res.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                f.write(chunk)
    return filename


def load_and_preview_roi(drive_id, custom_sections=None, mmap=False, preview=True):
    """Download an ROI stack from Google Drive, print its info, and preview it.

    drive_id         Google Drive file ID or share URL (file must be link-shared).
    custom_sections  list of section indices to preview (default: ~10 evenly spaced).
    mmap             keep the download on disk and memory-map it (huge stacks).
    preview          set False to skip the plot.

    Returns the ROI as a NumPy array.
    """
    m = re.search(r"[-\w]{25,}", drive_id)
    if not m:
        raise ValueError(f"Invalid Google Drive file ID or URL: '{drive_id}'")
    file_id = m.group(0)

    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    tmp.close()
    try:
        filename = _download_drive(file_id, tmp.name)
        size_mb = os.path.getsize(tmp.name) / 2**20
        roi = load_roi(tmp.name, mmap=mmap)
    except BaseException:
        os.remove(tmp.name)
        raise
    if mmap:
        print(f"Memory-mapped from {tmp.name}")
    else:
        try:
            os.remove(tmp.name)  # array is fully in RAM now
        except OSError:
            pass

    print(f"Filename : {filename}")
    print(f"Shape    : {roi.shape}")
    print(f"Dtype    : {roi.dtype}")
    print(f"Size     : {size_mb:.2f} MB")

    meta = _parse_filename(re.sub(r"\.npy$", "", filename, flags=re.I))
    if meta:
        print("\nParsed Metadata:")
        for k, v in meta.items():
            print(f"  • {k:<25}: {v}")

    if preview:
        n = _n_sections(roi)
        if custom_sections is not None:
            preview_sections(roi, sections=custom_sections)
        else:
            preview_sections(roi, sections=np.linspace(0, n - 1, min(10, n), dtype=int))
    return roi


# --------------------------------------------------------------------------- #
# Exploring
# --------------------------------------------------------------------------- #
def preview_sections(roi, start=None, end=None, step=1, sections=None,
                     max_sections=30, ncols=5):
    """Plot sections in a grid.

    preview_sections(roi, start=100, end=150)       sections 100..149 (end exclusive)
    preview_sections(roi, start=100, end=150, step=10)
    preview_sections(roi, sections=[10, 25, 80])    exactly these sections
    preview_sections(roi)                           ~10 evenly spaced sections

    Long ranges are thinned to max_sections evenly spaced ones to keep plots fast.
    """
    import matplotlib.pyplot as plt

    n = _n_sections(roi)
    if sections is None and start is None and end is None and step == 1:
        idx = list(np.unique(np.linspace(0, n - 1, min(10, n), dtype=int)))
    else:
        idx = _select(n, start, end, step, sections)
    if len(idx) > max_sections:
        print(f"Showing {max_sections} of {len(idx)} sections "
              f"(raise max_sections or use step= to change).")
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, max_sections, dtype=int)]

    ncols = max(1, min(ncols, len(idx)))
    nrows = -(-len(idx) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idx):
        _show(ax, roi[i] if roi.ndim >= 3 else roi)
        ax.set_title(f"Section {i}", fontsize=10)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def get_sections(roi, start=None, end=None, step=1, sections=None):
    """Return a sub-stack of sections.

    get_sections(roi, 100, 150)            sections 100..149  (a view, no copy)
    get_sections(roi, sections=[10, 25])   those sections     (a copy)
    """
    n = _n_sections(roi)
    if sections is not None:
        return roi[_select(n, sections=sections)]
    idx = _select(n, start, end, step)
    return roi[idx[0]:idx[-1] + 1:step]


def roi_info(roi, max_sections=20):
    """Print and return shape, dtype, size and intensity statistics.

    Statistics use up to max_sections evenly spaced sections so they stay fast
    on large or memory-mapped stacks.
    """
    n = _n_sections(roi)
    idx = np.unique(np.linspace(0, n - 1, min(max_sections, n), dtype=int))
    sample = roi[idx] if roi.ndim >= 3 else roi
    info = {
        "shape": tuple(roi.shape),
        "dtype": str(roi.dtype),
        "size_mb": round(roi.nbytes / 2**20, 2),
        "sections": n,
        "min": float(sample.min()),
        "max": float(sample.max()),
        "mean": float(sample.mean(dtype=np.float64)),
        "std": float(sample.std(dtype=np.float64)),
    }
    if np.issubdtype(roi.dtype, np.floating):
        info["nan_count"] = int(np.isnan(sample).sum())
    for k, v in info.items():
        print(f"{k:<10}: {v}")
    if len(idx) < n:
        print(f"(min/max/mean/std from {len(idx)} of {n} sections)")
    return info


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def _normalize_block(x, method, clip_percentiles):
    """Normalize float array x in place."""
    if clip_percentiles is not None:
        flat = x.reshape(-1)
        sample = flat[::max(1, flat.size // 1_000_000)]  # percentile on a subsample
        lo, hi = np.percentile(sample, clip_percentiles)
        np.clip(x, lo, hi, out=x)
    if method == "minmax":
        lo, hi = float(x.min()), float(x.max())
        x -= lo
        if hi > lo:
            x /= (hi - lo)
    else:  # zscore
        mean, std = float(x.mean(dtype=np.float64)), float(x.std(dtype=np.float64))
        x -= mean
        if std > 0:
            x /= std


def normalize_roi(roi, method="minmax", clip_percentiles=None, per_section=False,
                  dtype=np.float32, inplace=False):
    """Rescale intensities; returns a new float array (input is not modified).

    method            "minmax" -> values in [0, 1];  "zscore" -> mean 0, std 1.
    clip_percentiles  e.g. (1, 99) clips outliers first (robust to bright/dark artifacts).
    per_section       normalize each section on its own instead of the whole stack.
    inplace           reuse roi's memory (only if roi is already a writable float array).
    """
    if method not in ("minmax", "zscore"):
        raise ValueError("method must be 'minmax' or 'zscore'")
    if inplace:
        if not (np.issubdtype(roi.dtype, np.floating) and roi.flags.writeable):
            raise ValueError("inplace=True needs a writable float array")
        x = roi
    else:
        x = np.array(roi, dtype=dtype)  # the only copy
    if per_section and x.ndim >= 3:
        for i in range(x.shape[0]):
            _normalize_block(x[i], method, clip_percentiles)
    else:
        _normalize_block(x, method, clip_percentiles)
    return x


def crop_roi(roi, y=None, x=None):
    """Crop rows/columns of every section; returns a view (no copy).

    crop_roi(roi, y=(100, 300), x=(50, 250))   rows 100..299, columns 50..249
    """
    ys = slice(*y) if y is not None else slice(None)
    xs = slice(*x) if x is not None else slice(None)
    return roi[(slice(None), ys, xs) if roi.ndim >= 3 else (ys, xs)]


def center_crop(roi, size):
    """Crop the central size x size (or (h, w)) region of every section; a view."""
    h, w = (size, size) if np.isscalar(size) else size
    H, W = roi.shape[-3:-1] if _channels_last(roi) else roi.shape[-2:]
    if h > H or w > W:
        raise ValueError(f"Crop {(h, w)} is larger than the section {(H, W)}.")
    y0, x0 = (H - h) // 2, (W - w) // 2
    return crop_roi(roi, y=(y0, y0 + h), x=(x0, x0 + w))


def downsample_roi(roi, factor=2, method="mean"):
    """Shrink each section by an integer factor in height and width.

    method="mean"    average factor x factor blocks (smooth, float32 output).
    method="stride"  keep every factor-th pixel (instant view, but can alias).
    Edges that don't divide evenly are dropped.
    """
    f = int(factor)
    if f < 1:
        raise ValueError("factor must be >= 1")
    ya, xa = (1, 2) if roi.ndim >= 3 else (0, 1)
    if method == "stride":
        sl = [slice(None)] * roi.ndim
        sl[ya] = slice(None, None, f)
        sl[xa] = slice(None, None, f)
        return roi[tuple(sl)]
    if method != "mean":
        raise ValueError("method must be 'mean' or 'stride'")
    sl = [slice(None)] * roi.ndim
    sl[ya] = slice(0, roi.shape[ya] // f * f)
    sl[xa] = slice(0, roi.shape[xa] // f * f)
    r = roi[tuple(sl)]
    shape = list(r.shape)
    shape[ya:xa + 1] = [r.shape[ya] // f, f, r.shape[xa] // f, f]
    return r.reshape(shape).mean(axis=(ya + 1, xa + 2), dtype=np.float32)


def to_grayscale(roi):
    """Convert an RGB(A) stack (Z, H, W, 3|4) to grayscale (Z, H, W), float32.

    Uses standard luminance weights; alpha is ignored. Grayscale input is returned as is.
    """
    if not (roi.ndim == 4 and roi.shape[-1] in (3, 4)):
        return roi[..., 0] if roi.ndim == 4 else roi
    out = np.zeros(roi.shape[:-1], dtype=np.float32)
    for c, w in enumerate((0.299, 0.587, 0.114)):  # per channel: no big temporary
        out += w * roi[..., c]
    return out


# --------------------------------------------------------------------------- #
# ML
# --------------------------------------------------------------------------- #
def to_torch(roi, channels_first=True, dtype=None):
    """Convert a stack to a PyTorch tensor of shape (N, C, H, W).

    Grayscale (Z, H, W) gets a channel axis of size 1. channels_first=False keeps
    the (N, H, W, C) layout. Memory is shared with the array when possible
    (writable arrays, matching dtype); read-only memmaps are copied.
    Integer data is usually best passed through normalize_roi() first.
    """
    import torch

    a = roi
    if a.ndim == 2:
        a = a[None]
    if a.ndim == 3:
        a = a[..., None]
    if a.ndim != 4:
        raise ValueError(f"Expected a (Z, H, W) or (Z, H, W, C) stack, got {roi.shape}")
    if channels_first:
        a = a.transpose(0, 3, 1, 2)
    if not a.flags.writeable:
        a = np.array(a)  # torch can't wrap read-only memory
    t = torch.from_numpy(a)  # works on strided views too
    return t if dtype is None else t.to(dtype)


# --------------------------------------------------------------------------- #
# QC / exploration helpers (numpy + scipy.ndimage only)
# --------------------------------------------------------------------------- #
_OD_LUT = (-np.log10((np.arange(256, dtype=np.float32) + 1.0) / 256.0)).astype(np.float32)


def to_od(sec):
    """Optical density of one uint8 section, summed over channels -> (H, W) float32.

    White (255) maps to ~0. Uses a 256-entry lookup table, so it is much faster than log10 per pixel.
    """
    a = np.asarray(sec)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    od = _OD_LUT[a]
    return od.sum(-1) if od.ndim == 3 else od


def tissue_mask(sec, od_thr=0.03, sigma=1.0, min_frac=5e-4):
    """Boolean tissue mask for one section: smoothed summed-OD > od_thr, components < min_frac of the FOV removed.

    od_thr=0.03 sits just above the OD of clipped-white background (<= ~0.026), so it does not depend on
    how dark the stain is (a fixed 'mean < 245' cut under-counts pale stains by 5-20 %).
    """
    from scipy import ndimage as ndi

    od = to_od(sec)
    if sigma:
        od = ndi.gaussian_filter(od, sigma)
    m = od > od_thr
    lab, n = ndi.label(m)
    if n:
        keep = np.bincount(lab.ravel()) >= max(4, int(min_frac * m.size))
        keep[0] = False
        m = keep[lab]
    return m


def section_stats(roi, od_thr=0.03, sections=None):
    """Per-section QC features -> dict of 1-D arrays (one entry per section; wrap in pandas.DataFrame if wanted).

    Keys: section, coverage, fg_mean, fg_std (tissue luminance), n_comp, frag (share of tissue outside the largest
    component), bbox_h, bbox_w, cy, cx (centroid, fraction of FOV), border (fraction of the image border that is tissue).
    """
    from scipy import ndimage as ndi

    idx = range(len(roi)) if sections is None else sections
    keys = ["coverage", "fg_mean", "fg_std", "n_comp", "frag", "bbox_h", "bbox_w", "cy", "cx", "border"]
    out = {k: [] for k in keys}
    out["section"] = list(idx)
    for z in out["section"]:
        s = np.asarray(roi[z])
        m = tissue_mask(s, od_thr)
        n = int(m.sum())
        lum = s.mean(-1) if s.ndim == 3 else s
        lab, nc = ndi.label(m)
        areas = np.bincount(lab.ravel())[1:] if nc else np.zeros(1)
        if n:
            ys, xs = np.nonzero(m)
            vals = (lum[m].mean(), lum[m].std(), ys.max() - ys.min() + 1, xs.max() - xs.min() + 1, ys.mean() / m.shape[0], xs.mean() / m.shape[1])
        else:
            vals = (np.nan,) * 6
        ring = np.concatenate([m[0], m[-1], m[1:-1, 0], m[1:-1, -1]])
        row = dict(coverage=n / m.size, fg_mean=vals[0], fg_std=vals[1], n_comp=nc, frag=1 - areas.max() / areas.sum() if nc else 0.0,
                   bbox_h=vals[2], bbox_w=vals[3], cy=vals[4], cx=vals[5], border=float(ring.mean()))
        for k in keys:
            out[k].append(row[k])
    return {k: np.asarray(v) for k, v in out.items()}


def local_zscore(v, window=7, floor_frac=0.02):
    """Robust z of each value against a rolling-median expectation -> (z, expected, scale).

    Right for series with a smooth trend along Z (tissue coverage is a dome, so a global z-score flags the
    natural ends). scale = MAD of the residuals, floored at floor_frac * (P95 - P5) so flat series don't explode.
    """
    from scipy.ndimage import median_filter

    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    ref = median_filter(np.where(ok, v, np.nanmedian(v)), size=window, mode="nearest")
    res = v - ref
    scale = max(1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res))), floor_frac * (np.nanpercentile(v, 95) - np.nanpercentile(v, 5)), 1e-9)
    return res / scale, ref, scale


def flag_sections(stats, z_thr=4.0, blank=0.005, near_blank_rel=0.10, min_cov_change=0.02, frag_max=0.05, edge=5):
    """Rule-based section QC on section_stats() output -> list of (section, reason, value, (expected_lo, expected_hi)).

    Rules: blank (coverage < blank), nearly_blank (< near_blank_rel x P90 coverage), small/large_tissue (local z of
    coverage > z_thr and >= min_cov_change away from expectation), fragmented (frag > frag_max). Contiguous low-coverage
    runs at either end of the stack and the first/last `edge` sections are tagged 'series_end' instead of being
    reported as defects (tissue physically enters/leaves the FOV there).
    """
    cov, sec = stats["coverage"], stats["section"]
    z, ref, sc = local_zscore(cov)
    plateau = np.percentile(cov, 90)
    low = cov < near_blank_rel * plateau
    end = np.zeros(len(cov), bool)
    end[:edge] = end[-edge:] = True
    for rng in (range(len(cov)), range(len(cov) - 1, -1, -1)):
        for i in rng:
            if not low[i]:
                break
            end[i] = True
    out = []
    for i, s in enumerate(sec):
        tag = lambda r: ("series_end:" + r) if end[i] else r
        lo, hi = ref[i] - z_thr * sc, ref[i] + z_thr * sc
        if cov[i] < blank:
            out.append((s, tag("blank"), cov[i], (blank, 1.0)))
            continue
        if low[i]:
            out.append((s, tag("nearly_blank"), cov[i], (near_blank_rel * plateau, 1.0)))
        elif abs(cov[i] - ref[i]) >= min_cov_change and abs(z[i]) > z_thr:
            out.append((s, tag("small_tissue" if z[i] < 0 else "large_tissue"), cov[i], (lo, hi)))
        if stats["frag"][i] > frag_max:
            out.append((s, tag("fragmented"), stats["frag"][i], (0.0, frag_max)))
    return out


def lowres_stack(roi, size=64):
    """OD map of every section resampled to (size, size) by block averaging -> (Z, size, size) float32.

    Small, stain-agnostic (white = 0) representation for continuity checks and cross-stain comparison.
    """
    from scipy import ndimage as ndi

    out = np.empty((len(roi), size, size), np.float32)
    for z in range(len(roi)):
        od = to_od(roi[z])
        out[z] = ndi.zoom(ndi.uniform_filter(od, max(1, od.shape[0] // size)), (size / od.shape[0], size / od.shape[1]), order=1)
    return out


def adjacent_change(roi, size=64, lag=1):
    """Structural change between sections N and N+lag: 1 - Pearson NCC of low-res OD maps -> (Z-lag,) array.

    Blank-vs-blank = 0, blank-vs-content = 1. A single large value with small skip-distance d(N-1, N+1) means
    section N is a bad section; a persistent jump means a real discontinuity in the series.
    """
    lo = lowres_stack(roi, size).reshape(len(roi), -1).astype(np.float64)
    lo -= lo.mean(1, keepdims=True)
    nrm = np.linalg.norm(lo, axis=1)
    a, b = lo[:-lag], lo[lag:]
    na, nb = nrm[:-lag], nrm[lag:]
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = (a * b).sum(1) / (na * nb)
    ncc = np.where((na < 1e-9) & (nb < 1e-9), 1.0, np.where((na < 1e-9) | (nb < 1e-9), 0.0, ncc))
    return 1.0 - ncc
