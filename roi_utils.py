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

QC for tissue-only (background-removed) stacks:
    st = section_stats(roi)                  # per-section mean/std/percentiles/range/sharpness (+ background check)
    flags = flag_sections(st)                # unusual values along Z: [(section, reason, value, (lo, hi)), ...]
    an = find_section_anomalies(roi)         # isolated bad sections: dict with "section", "score", ...
    rows = resolve_atlas_sections(an["section"], biosample_id=580, stain="NISL",
                                  center_section=meta["center_section"], n_sections=len(roi))
    preview_sections(roi, sections=an["section"], context=1)   # each bad section with its neighbours

Fallback for sections that still contain background: tissue_mask(roi[i]) (optical-density based).

Requires numpy (scipy for the QC helpers, requests for the Atlas lookup). matplotlib (previews), requests
(Google Drive) and torch (to_torch) are imported only when those functions are used.
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
    "find_section_anomalies",
    "resolve_atlas_sections",
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


def load_and_preview_roi(drive_id, custom_sections=None, mmap=False, preview=True, return_meta=False):
    """Download an ROI stack from Google Drive, print its info, and preview it.

    drive_id         Google Drive file ID or share URL (file must be link-shared).
    custom_sections  list of section indices to preview (default: ~10 evenly spaced).
    mmap             keep the download on disk and memory-map it (huge stacks).
    preview          set False to skip the plot.
    return_meta      also return {"filename", "center_section", "stain"} parsed from the file name
                     (center_section is the Atlas section the stack is centred on; see resolve_atlas_sections).

    Returns the ROI as a NumPy array (or (roi, meta) with return_meta=True).
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
    if return_meta:
        se = re.search(r"(?:SE|se|sec|section(?:_number)?)[-_=]?(\d+)", re.sub(r"\.npy$", "", filename, flags=re.I))
        return roi, {"filename": filename, "center_section": int(se.group(1)) if se else None,
                     "stain": meta.get("Stain")}
    return roi


# --------------------------------------------------------------------------- #
# Exploring
# --------------------------------------------------------------------------- #
def preview_sections(roi, start=None, end=None, step=1, sections=None,
                     max_sections=30, ncols=5, context=0):
    """Plot sections in a grid.

    preview_sections(roi, start=100, end=150)       sections 100..149 (end exclusive)
    preview_sections(roi, start=100, end=150, step=10)
    preview_sections(roi, sections=[10, 25, 80])    exactly these sections
    preview_sections(roi)                           ~10 evenly spaced sections
    preview_sections(roi, sections=[54], context=1) section 54 plus 1 neighbour each side (54 in red)

    Long ranges are thinned to max_sections evenly spaced ones to keep plots fast.
    """
    import matplotlib.pyplot as plt

    n = _n_sections(roi)
    if sections is None and start is None and end is None and step == 1:
        idx = list(np.unique(np.linspace(0, n - 1, min(10, n), dtype=int)))
    else:
        idx = _select(n, start, end, step, sections)
    marked = set(idx) if context else set()
    if context:
        idx = sorted({j for i in idx for j in range(i - context, i + context + 1) if 0 <= j < n})
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
        ax.set_title(f"Section {i}", fontsize=10, color="red" if i in marked else "black")
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
# QC helpers (numpy + scipy.ndimage only)
#
# Primary QC reads the pixels as they are: stacks are normally already
# background-removed (tissue only), so no segmentation is run by default.
# to_od() / tissue_mask() are a FALLBACK for sections with residual background.
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
    """FALLBACK background removal for one section with a white background -> boolean tissue mask.

    Smoothed summed-OD > od_thr, components < min_frac of the FOV removed. od_thr=0.03 sits just above the OD
    of clipped-white background (<= ~0.026), so it does not depend on how dark the stain is. Only needed for
    sections that still contain background; section_stats() calls it only for those.
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


def _gray2d(sec):
    a = np.asarray(sec, dtype=np.float32)
    return a.mean(-1) if a.ndim == 3 else a


def _sharpness(g):
    """Variance of the Laplacian: low = blurry / out of focus."""
    if min(g.shape) < 3:
        return 0.0
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def _white_level(roi):
    """'Near-white' intensity for this dtype (0.96 of full scale)."""
    if np.issubdtype(roi.dtype, np.integer):
        return 0.96 * np.iinfo(roi.dtype).max
    return 0.96 if float(np.max(roi[0])) <= 1.0 else 245.0


def section_stats(roi, sections=None, bg_thr=0.05):
    """Per-section QC features for tissue-only stacks -> dict of 1-D arrays (wrap in pandas.DataFrame if wanted).

    Keys: section; mean, std, p1, p99, dynamic_range (p99 - p1), sharpness (variance of Laplacian);
    white_frac / zero_frac (share of near-white / all-zero pixels = residual background check);
    tissue_frac (1.0 when both are below bg_thr, i.e. no segmentation needed; otherwise measured with the
    background-removal fallback, so it is only meaningful where background remains).
    Works on (Z, H, W) or (Z, H, W, C); one section in memory at a time.
    """
    idx = list(range(_n_sections(roi))) if sections is None else [int(i) for i in np.atleast_1d(sections)]
    white = _white_level(roi)
    keys = ["mean", "std", "p1", "p99", "dynamic_range", "sharpness", "white_frac", "zero_frac", "tissue_frac"]
    out = {k: [] for k in keys}
    for z in idx:
        s = np.asarray(roi[z])
        g = _gray2d(s)
        sub = g.ravel()[::max(1, g.size // 250_000)]  # percentiles on a subsample
        p1, p99 = np.percentile(sub, (1, 99))
        wf = float((g >= white).mean())
        zf = float((s == 0).all(-1).mean() if s.ndim == 3 else (s == 0).mean())
        if max(wf, zf) < bg_thr:
            tf = 1.0
        elif zf >= wf:
            tf = 1.0 - zf  # black background: non-zero = tissue
        else:
            tf = float(tissue_mask(s * 255 if white < 1 else s).mean())
        row = dict(mean=g.mean(dtype=np.float64), std=g.std(dtype=np.float64), p1=p1, p99=p99, dynamic_range=p99 - p1,
                   sharpness=_sharpness(g), white_frac=wf, zero_frac=zf, tissue_frac=tf)
        for k in keys:
            out[k].append(row[k])
    res = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    res["section"] = np.asarray(idx)
    return res


def local_zscore(v, window=7, floor_frac=0.02):
    """Robust z of each value against a rolling-median expectation -> (z, expected, scale).

    Right for series with a smooth trend along Z (a global z-score would flag the natural ends).
    scale = MAD of the residuals, floored at floor_frac * (P95 - P5) so flat series don't explode.
    """
    from scipy.ndimage import median_filter

    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    ref = median_filter(np.where(ok, v, np.nanmedian(v)), size=window, mode="nearest")
    res = v - ref
    scale = max(1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res))), floor_frac * (np.nanpercentile(v, 95) - np.nanpercentile(v, 5)), 1e-9)
    return res / scale, ref, scale


def flag_sections(stats, z_thr=4.0, window=7, min_rel=0.05, bg_thr=0.05,
                  metrics=("mean", "std", "dynamic_range", "sharpness")):
    """Flag sections whose QC values are unusual along Z -> list of (section, reason, value, (expected_lo, expected_hi)).

    Each metric in `metrics` is compared with its own rolling-median trend (local_zscore), so there are no global
    thresholds; a deviation must also be >= min_rel of the expected value. Reasons look like 'sharpness_low'.
    'residual_background' marks sections with a clearly larger white/zero share (> bg_thr) than their neighbours;
    stack-wide background shows up directly in stats['white_frac'] / stats['zero_frac'].
    """
    sec, out = stats["section"], []
    for k in metrics:
        v = stats[k]
        z, ref, sc = local_zscore(v, window)
        bad = (np.abs(z) > z_thr) & (np.abs(v - ref) >= min_rel * np.abs(ref))
        for i in np.flatnonzero(bad):
            out.append((int(sec[i]), f"{k}_{'high' if z[i] > 0 else 'low'}", float(v[i]), (float(ref[i] - z_thr * sc), float(ref[i] + z_thr * sc))))
    bg = np.maximum(stats["white_frac"], stats["zero_frac"])
    _, bref, _ = local_zscore(bg, window)
    for i in np.flatnonzero((bg > bg_thr) & (bg - bref > bg_thr / 2)):
        out.append((int(sec[i]), "residual_background", float(bg[i]), (0.0, bg_thr)))
    return sorted(out, key=lambda r: r[0])


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


def adjacent_change(roi, size=64, lag=1, lowres=None):
    """Structural change between sections N and N+lag: 1 - Pearson NCC of low-res OD maps -> (Z-lag,) array.

    lag=1: normal neighbours; lag=2 (N-1 vs N+1) tells an isolated bad section from a real anatomical transition.
    Blank-vs-blank = 0, blank-vs-content = 1. Pass lowres=lowres_stack(roi) to reuse it across several calls.
    """
    lo = lowres_stack(roi, size) if lowres is None else lowres
    lo = lo.reshape(len(lo), -1).astype(np.float64)
    lo -= lo.mean(1, keepdims=True)
    nrm = np.linalg.norm(lo, axis=1)
    a, b = lo[:-lag], lo[lag:]
    na, nb = nrm[:-lag], nrm[lag:]
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = (a * b).sum(1) / (na * nb)
    ncc = np.where((na < 1e-9) & (nb < 1e-9), 1.0, np.where((na < 1e-9) | (nb < 1e-9), 0.0, ncc))
    return 1.0 - ncc


def find_section_anomalies(roi, size=64, z_thr=5.0, min_change=0.02, skip_ratio=0.5, lowres=None):
    """Find isolated bad sections -> dict of arrays sorted by score: section, d_prev, d_next, d_skip, score.

    Section N is suspicious when it differs from BOTH neighbours (d_prev = d1[N-1] and d_next = d1[N] are high:
    above the stack's median change by z_thr robust SDs, and at least min_change) while the neighbours still
    resemble each other (d_skip = d2[N-1] < skip_ratio * min(d_prev, d_next)). score = min(d_prev, d_next) - d_skip.
    The first and last section have one neighbour and are not tested; two adjacent bad sections are not caught.
    Returned `section` values are stack indices (use resolve_atlas_sections for Atlas numbers).
    """
    lo = lowres_stack(roi, size) if lowres is None else lowres
    keys = ("section", "d_prev", "d_next", "d_skip", "score")
    if len(lo) < 3:
        return {k: np.empty(0) for k in keys}
    d1 = adjacent_change(None, lag=1, lowres=lo)
    d2 = adjacent_change(None, lag=2, lowres=lo)
    med = np.median(d1)
    hi = med + max(z_thr * 1.4826 * np.median(np.abs(d1 - med)), min_change)
    prev_, next_ = d1[:-1], d1[1:]  # entry k describes section N = k + 1
    both = np.minimum(prev_, next_)
    hit = np.flatnonzero((prev_ > hi) & (next_ > hi) & (d2 < skip_ratio * both))
    order = hit[np.argsort(-(both[hit] - d2[hit]))]
    return {"section": order + 1, "d_prev": prev_[order], "d_next": next_[order], "d_skip": d2[order],
            "score": both[order] - d2[order]}


# --------------------------------------------------------------------------- #
# Atlas section numbers
# --------------------------------------------------------------------------- #
# Sections endpoint used by resolve_atlas_sections(). Change it here for production; its query string
# (biosample_id, stain) supplies the defaults when those arguments are omitted.
ATLAS_SECTIONS_URL = "http://172.20.23.183:8054/sections?biosample_id=580&stain=NISL"


def _section_number(rec, field=None):
    """Atlas section number from one /sections entry (a bare number, or a dict holding one)."""
    if isinstance(rec, (int, np.integer)) or (isinstance(rec, str) and rec.strip().isdigit()):
        return int(rec)
    if isinstance(rec, dict):
        for k in ((field,) if field else ("section", "section_number", "sectionNumber", "section_id", "se", "number", "id")):
            if k in rec:
                return int(rec[k])
        raise RuntimeError(f"Can't find the section number in record {rec}; pass section_field=<key>.")
    raise RuntimeError(f"Unexpected /sections entry: {rec!r}")


def _window_start(n_total, pos, n):
    """Start position of an n-section stack centred on list position pos (window clamped at the list ends)."""
    for z in range(n):
        start, end = max(0, pos - z), min(n_total, pos + z + 1)
        if end - start == n:
            return start
    raise ValueError(f"A stack of {n} sections can't come from a list of {n_total} sections.")


def resolve_atlas_sections(stack_indices, biosample_id=None, stain=None, center_section=None, n_sections=None,
                           first_section=None, api_url=None, section_field=None, timeout=10, verbose=True):
    """Map ROI-stack indices to Atlas section numbers with ONE request to the /sections endpoint.

    stack_indices   any list/array of indices into the stack (e.g. anomalies["section"]).
    biosample_id, stain   passed to the endpoint; default to the ones in ATLAS_SECTIONS_URL (580, "NISL").
    The stack is a contiguous window of the endpoint's ordered section list, so say where it starts with EITHER
      first_section   Atlas section number of stack index 0, or
      center_section + n_sections   the SE number in the file name (load_and_preview_roi(..., return_meta=True))
                      and len(roi); the window is centred on the nearest listed section.
    api_url         override ATLAS_SECTIONS_URL for one call, e.g. "http://host:8054" or a full /sections URL.
    section_field   key holding the number if the endpoint returns dict records instead of plain numbers.

    Returns a list of dicts: stack_index, biosample_id, stain, atlas_section, record (the API's own entry).
    Out-of-range indices give atlas_section=None. Raises RuntimeError if the API can't be reached or read.
    """
    import requests
    from urllib.parse import parse_qsl, urlsplit

    parts = urlsplit(api_url or ATLAS_SECTIONS_URL)
    params = dict(parse_qsl(parts.query))
    if biosample_id is not None:
        params["biosample_id"] = biosample_id
    if stain:
        params["stain"] = stain
    biosample_id, stain = params.get("biosample_id"), params.get("stain")
    if first_section is None and center_section is None:
        raise ValueError("Say where the stack starts: first_section=<Atlas section of index 0>, or "
                         "center_section=<SE number> with n_sections=len(roi).")
    url = f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"
    url = url if url.endswith("/sections") else url + "/sections"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"Could not read sections from {url}: {e}") from None
    if isinstance(data, dict):  # {"sections": [...]}, or any single list inside a dict
        lists = [v for v in data.values() if isinstance(v, list)]
        data = data["sections"] if isinstance(data.get("sections"), list) else (lists[0] if len(lists) == 1 else None)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Unexpected or empty /sections response from {url}.")
    nums = np.array([_section_number(rec, section_field) for rec in data])  # API order is kept

    if first_section is not None:
        hits = np.flatnonzero(nums == int(first_section))
        if not len(hits):
            raise ValueError(f"first_section={first_section} is not in the section list for this biosample/stain.")
        p0 = int(hits[0])
    else:
        if n_sections is None:
            raise ValueError("center_section needs n_sections=len(roi).")
        p0 = _window_start(len(nums), int(np.abs(nums - int(center_section)).argmin()), int(n_sections))

    rows, bad = [], []
    for i in (int(i) for i in np.atleast_1d(stack_indices)):
        ok = 0 <= i < len(nums) - p0 and (n_sections is None or i < n_sections)
        if not ok:
            bad.append(i)
        rows.append({"stack_index": i, "biosample_id": biosample_id, "stain": stain,
                     "atlas_section": int(nums[p0 + i]) if ok else None, "record": data[p0 + i] if ok else None})
    if verbose and rows:
        print(f"{'stack_index':<12}| {'biosample_id':<13}| {'stain':<6}| atlas_section")
        for r_ in rows:
            print(f"{r_['stack_index']:<12}| {str(r_['biosample_id']):<13}| {str(r_['stain'] or '-'):<6}| {r_['atlas_section']}")
    if bad:
        print(f"Warning: stack indices outside this stack were not resolved: {bad}")
    return rows
