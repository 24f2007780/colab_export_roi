"""annotations_utils - overlay Stroke histology annotations on NumPy ROI stacks (never modifies the stack).

Annotations come from the Stroke OpenAtlas JSON API (GeoJSON FeatureCollections of Polygons):

    {api_base}/{PROTEIN}/{BIOSAMPLE}/{STAIN}/{SECTION}/B-{BIOSAMPLE}_SE-{SECTION}_registered.json
    e.g. https://c-stroke.humanbrain.in/histology_detections/openAtlasJson/GFAP/313/IHCS/1193/B-313_SE-1193_registered.json

Typical use (next to roi_utils.py):
    from roi_utils import *
    from annotations_utils import *

    available = find_available_annotations(biosample_id=313, stain="IHCS", section_num=1193)   # which proteins exist?
    annotations = load_stroke_annotations(313, "IHCS", 1193)                                   # registered coordinates

    roi, meta = load_and_preview_roi(ROI_DRIVE_ID, return_meta=True)
    roi_annotations = load_roi_annotations(roi, meta, biosample_id=313)    # whole stack, clipped to the ROI
    roi_annotations[54]["GFAP"]                                            # records for stack section 54
    preview_annotations(roi, roi_annotations, section_index=54, proteins=["GFAP", "CD68"])
    overlay = create_annotation_overlay(roi, roi_annotations, section_index=54)   # independent RGB array
    masks = create_annotation_masks(roi, roi_annotations, section_index=54)       # {protein: bool (H, W)}

`roi` is only ever READ. Overlays/masks are new arrays; annotation coordinates are never edited in place
(transformed geometry goes into a separate `roi_geometry` field next to the untouched `geometry`).

--------------------------------------------------------------------------------------------------------------
COORDINATE CONVENTION
--------------------------------------------------------------------------------------------------------------
Stroke JSON ("registered" / OpenAtlas space), observed in the real GFAP/313/IHCS/1193 file:
  * +X points right, and X is >= 0.
  * Y is NEGATIVE (-143764 ... -56148): the registered space is Y-up with the origin at the top of the section,
    so image rows grow with -Y. The registered Y axis is therefore inverted relative to image rows:
        row_in_section = -Y        (y_sign = -1)
  * The values are NOT pixels of the ROI and not even of the extractor's full-resolution grid: they reach
    x ~ 170,000 while that grid (legacy level 0 of the zarr store, checked on biosample 585 IHCS) is 24,000 px wide.
    Only 8 registered units per full-resolution pixel makes the data fit on both axes
    (x -> 480..21,200 px, |y| -> 7,000..18,000 px; 1:1 or 4:1 would overflow, 16:1 would leave the tissue in the
    left 44% of the section). That factor is DEFAULT_UNITS_PER_FULLRES_PX = 8 and is INFERRED, not published by
    the API: confirm it on one landmark (annotation_bounds() prints the implied pixel extent) and override with
    units_per_fullres_px= if your section grid differs. No section image for biosample 313 was reachable when
    this was written, so the factor could not be checked against pixels.

ROI side (taken from the ROI extractor, roi_extraction/apps/src/main.py):
  * center_X / center_Y (file name: X<..>_Y<..>) are on the extractor's FULL-RESOLUTION grid, +X right, +Y DOWN,
    origin at the section's top-left. They are converted to the ROI pyramid level (file name: L<n>) by dividing by the
    level factor (default 2**n, override with level_factor=), integer-truncated like the extractor does.
  * The ROI is a W x H window whose top-left corner is  (cx - W//2, cy - H//2)  in level pixels, with
    (cx, cy) = int(center / level_factor). ROI pixel (0, 0) is that top-left corner; columns grow with +X, rows
    grow DOWN. Continuous coordinates are used: pixel i covers [i, i+1), so imshow needs extent=(0, W, H, 0).
  * Windows touching the section edge are clamped by the extractor, so the centre is no longer the middle of the
    ROI. This is detected when the ROI is smaller than the size in the file name (a warning is raised); give
    origin_xy=(x0, y0) (level pixels of ROI pixel (0, 0)) to fix it.

Full transform (all in float):
    col = X / (units_per_fullres_px * level_factor) - x0
    row = (-Y) / (units_per_fullres_px * level_factor) - y0

Requires numpy, requests, shapely (clipping), Pillow (masks/overlays); matplotlib only for preview_annotations.
Atlas section numbers come from roi_utils.resolve_atlas_sections (roi_utils.py must be importable).
"""

import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import quote

import numpy as np

__all__ = [
    "STROKE_PROTEINS",
    "STROKE_API_BASE",
    "DEFAULT_UNITS_PER_FULLRES_PX",
    "StrokeAnnotationError",
    "RoiTransform",
    "RoiAnnotations",
    "find_available_annotations",
    "load_stroke_annotations",
    "make_roi_transform",
    "to_roi_coordinates",
    "load_roi_annotations",
    "annotation_bounds",
    "create_annotation_masks",
    "create_annotation_overlay",
    "preview_annotations",
]

STROKE_API_BASE = "https://c-stroke.humanbrain.in/histology_detections/openAtlasJson"

# Candidate annotation types (protein code -> description). Not every one exists for every section.
STROKE_PROTEINS = {
    "CD68": "Microglial density (CD68+)",
    "GFAP": "Astrocyte density (GFAP+)",
    "Fib": "BBB disruption (Fib+)",
    "CD34": "Vessel density (CD34+)",
    "HIF1a": "Hypoxic regions (HIF1a+)",
    "Nissl": "Cell density (Nissl+)",
    "H&E": "Infarct (H&E)",
    "APP": "Axonal injury (APP+)",
    "GAP43": "Axonal regrowth (GAP43+)",
}

# Registered-coordinate units per full-resolution ROI-extractor pixel. INFERRED (see module docstring): verify it.
DEFAULT_UNITS_PER_FULLRES_PX = 8.0

_COLORS = {
    "CD68": "#D55E00", "GFAP": "#009E73", "Fib": "#CC79A7", "CD34": "#0072B2", "HIF1a": "#F0E442",
    "Nissl": "#56B4E9", "H&E": "#7F7F7F", "APP": "#8E44AD", "GAP43": "#FF7F00",
}

_COLLECTION_CACHE = {}  # url -> parsed FeatureCollection; never mutated, only read


class StrokeAnnotationError(RuntimeError):
    """Network / HTTP / format problem while talking to the Stroke annotation API."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _check_proteins(proteins):
    """Validate protein codes (case-insensitive) and return them in canonical spelling, in candidate order."""
    if proteins is None:
        return list(STROKE_PROTEINS)
    if isinstance(proteins, str):
        proteins = [proteins]
    canon = {p.lower(): p for p in STROKE_PROTEINS}
    out = []
    for p in proteins:
        c = canon.get(str(p).lower())
        if c is None:
            raise ValueError(f"Unknown protein {p!r}. Supported: {', '.join(STROKE_PROTEINS)}.")
        if c not in out:
            out.append(c)
    return out


def stroke_url(biosample_id, stain, section_num, protein, api_base=STROKE_API_BASE):
    """URL of one annotation file (the protein is percent-encoded, e.g. 'H&E' -> 'H%26E')."""
    return (f"{api_base.rstrip('/')}/{quote(protein, safe='')}/{biosample_id}/{stain}/{section_num}/"
            f"B-{biosample_id}_SE-{section_num}_registered.json")


def _session(workers):
    import requests
    from requests.adapters import HTTPAdapter

    s = requests.Session()
    s.mount("https://", HTTPAdapter(pool_connections=workers, pool_maxsize=workers))
    s.mount("http://", HTTPAdapter(pool_connections=workers, pool_maxsize=workers))
    return s


def _probe(session, url, timeout):
    """Is there a real JSON annotation at `url`? -> dict(available, status, error).

    The server answers HTTP 200 with its HTML single-page app for annotations that do not exist (verified for
    CD68/CD34/... on 313/IHCS/1193), so the status code alone cannot be trusted: the body type decides.
    404 simply means 'not available'. Anything else unexpected is an error (reported, never hidden).
    Only headers (HEAD) are fetched; no geometry is downloaded here.
    """
    import requests

    try:
        r = session.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code in (405, 501):  # HEAD not supported: fetch just the first bytes
            r.close()
            r = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
        status = r.status_code
        ctype = r.headers.get("Content-Type", "").lower()
        if status == 404:
            return {"available": False, "status": 404, "error": None}
        if status != 200:
            return {"available": False, "status": status, "error": f"HTTP {status}"}
        if "json" in ctype:
            return {"available": True, "status": 200, "error": None}
        if "html" in ctype:  # SPA fallback page = no such annotation
            return {"available": False, "status": 200, "error": None}
        # Unknown / missing content type: look at the first byte of the body
        g = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
        try:
            head = next(g.iter_content(chunk_size=64), b"").lstrip()
        finally:
            g.close()
        return {"available": head[:1] in (b"{", b"["), "status": 200, "error": None}
    except requests.RequestException as e:
        return {"available": False, "status": None, "error": f"{type(e).__name__}: {e}"}


def _probe_many(session, urls, timeout, workers):
    urls = list(dict.fromkeys(urls))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(urls) or 1))) as ex:
        return dict(zip(urls, ex.map(lambda u: _probe(session, u, timeout), urls)))


def _fetch_collection(session, url, timeout, use_cache=True):
    """Download and parse one GeoJSON FeatureCollection (raises StrokeAnnotationError if it is not one)."""
    import requests

    if use_cache and url in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[url]
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise StrokeAnnotationError(f"Could not download {url}: {e}") from None
    except ValueError:
        raise StrokeAnnotationError(f"{url} did not return JSON (not an annotation file).") from None
    if not (isinstance(data, dict) and data.get("type") == "FeatureCollection"
            and isinstance(data.get("features"), list)):
        raise StrokeAnnotationError(f"{url} is not a GeoJSON FeatureCollection.")
    if use_cache:
        _COLLECTION_CACHE[url] = data
    return data


def _warn_errors(rows):
    errs = [r for r in rows if r.get("error")]
    if errs:
        warnings.warn("Availability could not be determined for: " + "; ".join(
            f"{r['protein']} ({r['error']})" for r in errs), stacklevel=3)


# --------------------------------------------------------------------------- #
# 1. Availability discovery
# --------------------------------------------------------------------------- #
def find_available_annotations(biosample_id, stain, section_num, proteins=None, api_base=STROKE_API_BASE,
                               timeout=20, workers=9, verbose=False):
    """Check which annotation types really exist for one section (headers only, geometry is not downloaded).

    Returns a list of dicts, one per candidate protein, in this order:
        protein | description | available | url | status | error
    `available` is True only for a real JSON annotation. A missing annotation (HTTP 404, or the server's HTML
    fallback page) is simply available=False with error=None; real HTTP/network problems fill `error` (and warn).
    Wrap it in pandas.DataFrame(...) to display it. verbose=True also prints a table.
    """
    plist = _check_proteins(proteins)
    urls = {p: stroke_url(biosample_id, stain, section_num, p, api_base) for p in plist}
    with _session(workers) as s:
        res = _probe_many(s, urls.values(), timeout, workers)
    rows = [{"protein": p, "description": STROKE_PROTEINS[p], "available": res[urls[p]]["available"],
             "url": urls[p], "status": res[urls[p]]["status"], "error": res[urls[p]]["error"]} for p in plist]
    _warn_errors(rows)
    if verbose:
        print(f"{'protein':<8}| {'description':<27}| {'available':<10}| url")
        for r in rows:
            print(f"{r['protein']:<8}| {r['description']:<27}| {str(r['available']):<10}| {r['url']}"
                  + (f"   !! {r['error']}" if r["error"] else ""))
    return rows


# --------------------------------------------------------------------------- #
# 2. Loading (registered coordinates, untouched)
# --------------------------------------------------------------------------- #
def _records_from_collection(coll, protein, url, biosample_id, stain, section_num):
    recs, skipped = [], 0
    for i, feat in enumerate(coll["features"]):
        geom = feat.get("geometry") if isinstance(feat, dict) else None
        if not geom or geom.get("type") not in ("Polygon", "MultiPolygon") or not geom.get("coordinates"):
            skipped += 1
            continue
        recs.append({
            "protein": protein,
            "description": STROKE_PROTEINS[protein],
            "biosample_id": biosample_id,
            "stain": stain,
            "section": section_num,
            "feature_index": i,
            "feature_id": feat.get("id"),
            "geometry": geom,                      # ORIGINAL registered coordinates, never edited
            "properties": feat.get("properties"),  # full GeoJSON properties
            "source_url": url,
            "collection_rotation": coll.get("rotation"),
            "roi_geometry": None,                  # filled by to_roi_coordinates()
        })
    if skipped:
        warnings.warn(f"{protein}: skipped {skipped} feature(s) without Polygon/MultiPolygon geometry in {url}",
                      stacklevel=3)
    return recs


def load_stroke_annotations(biosample_id, stain, section_num, proteins=None, api_base=STROKE_API_BASE,
                            timeout=60, workers=9, use_cache=True):
    """Download the annotations of ONE section -> flat list of records (registered coordinates).

    proteins=None: discover availability first, then download only the available ones.
    proteins=[...]: those only; ones that do not exist are skipped with a warning (not an error).
    Each record: protein, description, biosample_id, stain, section, feature_index, feature_id, geometry (original
    GeoJSON geometry in registered coordinates), properties (full GeoJSON properties), source_url,
    collection_rotation, roi_geometry (None until to_roi_coordinates()).
    """
    plist = _check_proteins(proteins)
    avail = find_available_annotations(biosample_id, stain, section_num, plist, api_base, workers=workers)
    missing = [r["protein"] for r in avail if not r["available"] and not r["error"]]
    if proteins is not None and missing:
        warnings.warn(f"Not available for biosample {biosample_id} {stain} section {section_num}: {missing}",
                      stacklevel=2)
    rows = [r for r in avail if r["available"]]
    records = []
    with _session(workers) as s:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(rows) or 1))) as ex:
            colls = list(ex.map(lambda r: _fetch_collection(s, r["url"], timeout, use_cache), rows))
    for r, coll in zip(rows, colls):
        records += _records_from_collection(coll, r["protein"], r["url"], biosample_id, stain, section_num)
    return records


# --------------------------------------------------------------------------- #
# 3. Registered -> ROI coordinates
# --------------------------------------------------------------------------- #
@dataclass
class RoiTransform:
    """Registered (OpenAtlas) -> ROI pixel transform. See the module docstring for the convention.

    col = X / units_per_px - x0 ;  row = y_sign * Y / units_per_px - y0
    with units_per_px = units_per_fullres_px * level_factor.
    """
    units_per_px: float
    x0: float
    y0: float
    width: int
    height: int
    y_sign: int = -1
    units_per_fullres_px: float = DEFAULT_UNITS_PER_FULLRES_PX
    level_factor: float = 1.0
    center_xy: tuple = None

    def apply(self, xy):
        """(N, 2) registered coordinates -> (N, 2) ROI (col, row) coordinates, as a new array."""
        a = np.asarray(xy, dtype=float)
        out = np.empty_like(a[:, :2])
        out[:, 0] = a[:, 0] / self.units_per_px - self.x0
        out[:, 1] = (self.y_sign * a[:, 1]) / self.units_per_px - self.y0
        return out

    def describe(self):
        return (f"ROI {self.width}x{self.height}px, origin (level px) = ({self.x0:g}, {self.y0:g}), "
                f"level_factor={self.level_factor:g}, {self.units_per_fullres_px:g} registered units per full-res px "
                f"-> {self.units_per_px:g} units per ROI px, y_sign={self.y_sign}")


def _meta_get(meta, keys):
    for k in keys:
        if meta and meta.get(k) is not None:
            return meta[k]
    return None


def make_roi_transform(roi, meta=None, units_per_fullres_px=DEFAULT_UNITS_PER_FULLRES_PX, y_sign=-1,
                       level_factor=None, center_xy=None, origin_xy=None, size=None):
    """Build the RoiTransform for an ROI stack from its metadata.

    center_X / center_Y are taken from meta ("center_X"/"center_Y", "Center X"/"Center Y"), else parsed from
    meta["filename"] (X<n>_Y<n>); the level from "_L<n>" and the requested size from "_<w>x<h>". Override any of them:
      center_xy       (X, Y) on the full-resolution grid.
      level_factor    full-res px per ROI px (default 2**level from the file name, 1 if there is no level).
      origin_xy       (x0, y0): level-pixel position of ROI pixel (0, 0); skips the centre-based estimate (use it
                      for ROIs clamped at the section edge).
      size            requested ROI size, only used to detect clamping.
    """
    meta = meta or {}
    fname = str(meta.get("filename") or "")
    if roi.ndim < 3:
        raise ValueError("roi must be a (Z, H, W[, C]) stack")
    H, W = roi.shape[1], roi.shape[2]

    if center_xy is None:
        cx = _meta_get(meta, ("center_X", "center_x", "Center X"))
        cy = _meta_get(meta, ("center_Y", "center_y", "Center Y"))
        if cx is None:
            m = re.search(r"(?:^|[-_&])[xX][-_=]?(\d+)", fname)
            cx = m.group(1) if m else None
        if cy is None:
            m = re.search(r"(?:^|[-_&])[yY][-_=]?(\d+)", fname)
            cy = m.group(1) if m else None
        center_xy = (float(cx), float(cy)) if cx is not None and cy is not None else None
    if level_factor is None:
        lvl = _meta_get(meta, ("level", "Level"))
        if lvl is None:
            m = re.search(r"(?:^|[-_&])L(\d+)", fname)
            lvl = m.group(1) if m else 0
        level_factor = 2.0 ** int(lvl)
    if size is None:
        m = re.search(r"(\d+)x(\d+)", fname)
        size = (int(m.group(1)), int(m.group(2))) if m else None

    if origin_xy is None:
        if center_xy is None:
            raise ValueError("Could not find center_X/center_Y in meta (or its filename). "
                             "Pass center_xy=(X, Y) or origin_xy=(x0, y0).")
        cxl, cyl = int(center_xy[0] / level_factor), int(center_xy[1] / level_factor)  # truncation as in the extractor
        origin_xy = (cxl - W // 2, cyl - H // 2)
        if size is not None and (W, H) != tuple(size):
            warnings.warn(f"ROI is {W}x{H} but was requested as {size[0]}x{size[1]}: the window was clamped at the "
                          f"section edge, so its origin cannot be recovered from the centre. Pass origin_xy=(x0, y0).",
                          stacklevel=2)
    return RoiTransform(units_per_px=float(units_per_fullres_px) * float(level_factor), x0=float(origin_xy[0]),
                        y0=float(origin_xy[1]), width=int(W), height=int(H), y_sign=int(y_sign),
                        units_per_fullres_px=float(units_per_fullres_px), level_factor=float(level_factor),
                        center_xy=tuple(center_xy) if center_xy is not None else None)


def _make_valid(poly):
    try:
        from shapely.validation import make_valid
        return make_valid(poly)
    except ImportError:  # old shapely
        return poly.buffer(0)


def _polygons_of(g):
    """Polygon parts of any shapely result (Polygon / MultiPolygon / GeometryCollection / empty)."""
    t = g.geom_type
    if g.is_empty:
        return []
    if t == "Polygon":
        return [g]
    if t in ("MultiPolygon", "GeometryCollection"):
        return [p for sub in g.geoms for p in _polygons_of(sub)]
    return []


def _as_lists(x):
    if isinstance(x, (tuple, list)):
        return list(x) if x and isinstance(x[0], (int, float)) else [_as_lists(i) for i in x]
    return x


def _clip_geometry(geom, T):
    """GeoJSON Polygon/MultiPolygon in registered coordinates -> clipped GeoJSON in ROI pixels (None if outside)."""
    from shapely.geometry import MultiPolygon, Polygon, box, mapping
    from shapely.ops import unary_union

    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    frame = box(0, 0, T.width, T.height)
    parts = []
    for rings in polys:
        if not rings:
            continue
        ext = T.apply(np.asarray(rings[0], dtype=float))
        if (len(ext) < 3 or ext[:, 0].max() < 0 or ext[:, 0].min() > T.width
                or ext[:, 1].max() < 0 or ext[:, 1].min() > T.height):
            continue  # cheap reject: bounding box misses the ROI
        holes = [h for h in (T.apply(np.asarray(r, dtype=float)) for r in rings[1:]) if len(h) >= 3]
        poly = Polygon(ext, holes)
        if not poly.is_valid:
            poly = _make_valid(poly)
        parts += [p for p in _polygons_of(poly.intersection(frame)) if p.area > 0]
    if not parts:
        return None
    parts = _polygons_of(unary_union(parts)) if len(parts) > 1 else parts
    g = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    m = mapping(g)
    return {"type": m["type"], "coordinates": _as_lists(m["coordinates"])}


def to_roi_coordinates(annotations, roi, meta=None, transform=None, **transform_kwargs):
    """Registered-coordinate records -> NEW records clipped to the ROI, with `roi_geometry` filled in.

    Records whose polygons miss the ROI are dropped; partially inside polygons are clipped to the ROI rectangle
    [0, W] x [0, H]. The input records and their `geometry` are not modified. `transform` or transform_kwargs
    (see make_roi_transform) choose the transform; by default it is built from roi + meta.
    """
    T = transform or make_roi_transform(roi, meta, **transform_kwargs)
    out = []
    for rec in annotations:
        if rec.get("collection_rotation") not in (None, 0):
            warnings.warn(f"{rec['protein']}: annotation file has rotation={rec['collection_rotation']}, "
                          f"which is not applied.", stacklevel=2)
        g = _clip_geometry(rec["geometry"], T)
        if g is not None:
            new = dict(rec)
            new["roi_geometry"] = g
            out.append(new)
    return out


def annotation_bounds(annotations, units_per_fullres_px=DEFAULT_UNITS_PER_FULLRES_PX):
    """Extent of records' registered geometry, plus the full-resolution pixel extent it implies.

    Use it to check the scale: the implied pixel extent must fit inside the section (e.g. 24000 x 24000).
    """
    xy = [np.asarray(ring, dtype=float)[:, :2] for rec in annotations
          for poly in ([rec["geometry"]["coordinates"]] if rec["geometry"]["type"] == "Polygon"
                       else rec["geometry"]["coordinates"]) for ring in poly[:1]]
    if not xy:
        return None
    a = np.concatenate(xy)
    lo, hi = a.min(0), a.max(0)
    return {"registered_x": (float(lo[0]), float(hi[0])), "registered_y": (float(lo[1]), float(hi[1])),
            "fullres_px_x": (float(lo[0] / units_per_fullres_px), float(hi[0] / units_per_fullres_px)),
            "fullres_px_y": (float(abs(hi[1]) / units_per_fullres_px), float(abs(lo[1]) / units_per_fullres_px))}


# --------------------------------------------------------------------------- #
# 4. Whole-stack loading
# --------------------------------------------------------------------------- #
class RoiAnnotations(dict):
    """{stack_index: {protein: [record, ...]}} plus context. Behaves like that dict.

    roi_annotations[i]            -> {protein: records} for stack section i ({} if no annotation file exists)
    roi_annotations[i]["GFAP"]    -> records of GFAP that intersect the ROI (KeyError = GFAP does not exist for that
                                     section; [] = it exists but nothing falls inside the ROI)
    Attributes: atlas_sections {stack_index: Atlas section number}, availability {stack_index: rows from
    find_available_annotations}, transform (RoiTransform), biosample_id, stain.
    """
    atlas_sections = None
    availability = None
    transform = None
    biosample_id = None
    stain = None

    def records(self, section_index=None, proteins=None):
        """Flat list of records (one section, or all)."""
        idx = [section_index] if section_index is not None else sorted(self)
        want = None if proteins is None else _check_proteins(proteins)
        return [r for i in idx for p, rs in self.get(i, {}).items() if want is None or p in want for r in rs]


def load_roi_annotations(roi, meta, biosample_id, proteins=None, api_base=STROKE_API_BASE, stain=None,
                         section_indices=None, atlas_sections=None, atlas_api_url=None, timeout=60, workers=12,
                         verbose=True, **transform_kwargs):
    """Annotations for every section of an ROI stack, transformed to ROI pixels and clipped to the ROI.

    Steps: stack index -> Atlas section (roi_utils.resolve_atlas_sections, never assumed equal) -> availability per
    protein (headers only) -> download only what exists -> registered -> ROI coordinates -> clip. `roi` is not modified.

    roi, meta       stack and the dict from load_and_preview_roi(..., return_meta=True).
    biosample_id    Stroke biosample (e.g. 313).
    proteins        subset of STROKE_PROTEINS (default: whatever exists, per section).
    stain           default meta["stain"].
    section_indices stack indices to load (default: all).
    atlas_sections  explicit {stack_index: Atlas section} or list (one per stack index), used INSTEAD of the
                    /sections lookup (e.g. when that endpoint does not know the biosample).
    transform_kwargs  passed to make_roi_transform (units_per_fullres_px, level_factor, center_xy, origin_xy, ...).

    Returns a RoiAnnotations: roi_annotations[stack_index][protein] -> list of records (see load_stroke_annotations),
    each with `geometry` (registered, untouched) and `roi_geometry` (ROI pixels, clipped).
    """
    n = roi.shape[0]
    stain = stain or (meta or {}).get("stain")
    if not stain:
        raise ValueError("No stain: pass stain='IHCS' (meta['stain'] is empty).")
    plist = _check_proteins(proteins)
    idxs = list(range(n)) if section_indices is None else [int(i) for i in section_indices]
    bad = [i for i in idxs if not 0 <= i < n]
    if bad:
        raise IndexError(f"section_indices {bad} outside this stack (0..{n - 1}).")
    T = make_roi_transform(roi, meta, **transform_kwargs)

    # 1. stack index -> Atlas section
    if atlas_sections is not None:
        amap = dict(enumerate(atlas_sections)) if not isinstance(atlas_sections, dict) else dict(atlas_sections)
        missing = [i for i in idxs if i not in amap]
        if missing:
            raise ValueError(f"atlas_sections has no entry for stack indices {missing}.")
        atlas = {i: int(amap[i]) for i in idxs}
    else:
        from roi_utils import resolve_atlas_sections

        rows = resolve_atlas_sections(idxs, biosample_id=biosample_id, stain=stain,
                                      center_section=(meta or {}).get("center_section"), n_sections=n,
                                      api_url=atlas_api_url, verbose=False)
        atlas = {r["stack_index"]: r["atlas_section"] for r in rows if r["atlas_section"] is not None}

    # 2. availability for every (section, protein), in parallel; geometry is not touched yet
    urls = {(i, p): stroke_url(biosample_id, stain, atlas[i], p, api_base) for i in atlas for p in plist}
    with _session(workers) as s:
        probes = _probe_many(s, urls.values(), min(timeout, 30), workers)
        availability = {i: [{"protein": p, "description": STROKE_PROTEINS[p], "stack_index": i,
                             "atlas_section": atlas[i], "available": probes[urls[i, p]]["available"],
                             "url": urls[i, p], "status": probes[urls[i, p]]["status"],
                             "error": probes[urls[i, p]]["error"]} for p in plist] for i in atlas}
        _warn_errors([r for rows_ in availability.values() for r in rows_])

        # 3. download only what exists (each file once)
        need = list(dict.fromkeys(r["url"] for rows_ in availability.values() for r in rows_ if r["available"]))
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(need) or 1))) as ex:
            colls = dict(zip(need, ex.map(lambda u: _fetch_collection(s, u, timeout), need)))

    # 4. transform + clip
    out = RoiAnnotations()
    out.atlas_sections, out.availability, out.transform = atlas, availability, T
    out.biosample_id, out.stain = biosample_id, stain
    n_total = n_hit = 0
    for i in idxs:
        per = {}
        for r in availability.get(i, []):
            if not r["available"]:
                continue
            recs = _records_from_collection(colls[r["url"]], r["protein"], r["url"], biosample_id, stain,
                                            atlas[i])
            n_total += len(recs)
            per[r["protein"]] = to_roi_coordinates(recs, roi, transform=T)
            n_hit += len(per[r["protein"]])
        out[i] = per
    if verbose:
        _print_summary(out, idxs, T)
    if n_total and not n_hit:
        warnings.warn("Annotations exist but none intersects the ROI. Check center_X/center_Y and "
                      "units_per_fullres_px (see annotation_bounds()).", stacklevel=2)
    return out


def _print_summary(out, idxs, T):
    print(T.describe())
    withann = [i for i in idxs if any(out[i].values())]
    print(f"{len(idxs)} stack sections, {sum(1 for i in idxs if out[i])} with annotation files, "
          f"{len(withann)} with polygons inside the ROI")
    tot = {}
    for i in idxs:
        for p, rs in out[i].items():
            t = tot.setdefault(p, [0, 0])
            t[0] += 1
            t[1] += len(rs)
    for p, (ns, nf) in tot.items():
        print(f"  {p:<6} {STROKE_PROTEINS[p]:<27} in {ns} sections, {nf} polygons in the ROI")


# --------------------------------------------------------------------------- #
# 5. Masks / overlays / preview (all independent of roi)
# --------------------------------------------------------------------------- #
def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _color(rec, color_by):
    if color_by == "feature":
        d = (rec.get("properties") or {}).get("data") or {}
        c = d.get("color_hex_triplet")
        if isinstance(c, str) and re.fullmatch(r"#?[0-9a-fA-F]{6}", c):
            return _hex_rgb(c)
    return _hex_rgb(_COLORS[rec["protein"]])


def _records_for(annotations, section_index, proteins):
    want = None if proteins is None else _check_proteins(proteins)
    if isinstance(annotations, dict):
        if section_index not in annotations:
            raise KeyError(f"Stack section {section_index} was not loaded "
                           f"(loaded: {min(annotations, default=None)}..{max(annotations, default=None)}).")
        recs = [r for rs in annotations[section_index].values() for r in rs]
    else:
        recs = list(annotations)
        if any(r.get("roi_geometry") is None for r in recs):
            raise ValueError("These records are still in registered coordinates: run "
                             "to_roi_coordinates(records, roi, meta) first (or use load_roi_annotations).")
    return [r for r in recs if want is None or r["protein"] in want]


def _polys(g):
    return [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]


def _section_rgb(roi, section_index):
    """One section as a NEW (H, W, 3) uint8 array."""
    a = np.asarray(roi[section_index])
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = a[..., :3]
    if a.dtype == np.uint8:
        return a.copy()
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if 0.0 <= lo and hi <= 1.0:
        a = a * 255.0
    elif hi > lo:
        a = (a - lo) / (hi - lo) * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def create_annotation_masks(roi, annotations, section_index, proteins=None):
    """{protein: bool (H, W) mask} for one stack section. New arrays; `roi` is only used for its H, W."""
    from PIL import Image, ImageDraw

    H, W = roi.shape[1], roi.shape[2]
    masks = {}
    for rec in _records_for(annotations, section_index, proteins):
        im = masks.setdefault(rec["protein"], Image.new("L", (W, H), 0))
        d = ImageDraw.Draw(im)
        for rings in _polys(rec["roi_geometry"]):
            d.polygon([tuple(p) for p in rings[0]], fill=1)
            for hole in rings[1:]:
                d.polygon([tuple(p) for p in hole], fill=0)
    return {p: np.asarray(im, dtype=bool) for p, im in masks.items()}


def create_annotation_overlay(roi, annotations, section_index, proteins=None, alpha=0.4, outline=True,
                              color_by="protein"):
    """Section `section_index` with the annotation polygons blended on top -> NEW (H, W, 3) uint8 array.

    color_by="protein" (one colour per protein) or "feature" (each polygon's own color_hex_triplet from its
    GeoJSON properties, falling back to the protein colour). `roi` is not modified.
    """
    from PIL import Image, ImageDraw

    base = Image.fromarray(_section_rgb(roi, section_index)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(round(255 * alpha))
    for rec in _records_for(annotations, section_index, proteins):
        rgb = _color(rec, color_by)
        for rings in _polys(rec["roi_geometry"]):
            d.polygon([tuple(p) for p in rings[0]], fill=rgb + (a,))
            for hole in rings[1:]:
                d.polygon([tuple(p) for p in hole], fill=(0, 0, 0, 0))
            if outline:
                for ring in rings:
                    d.line([tuple(p) for p in ring], fill=rgb + (255,), width=1)
    return np.asarray(Image.alpha_composite(base, layer))[..., :3].copy()


def preview_annotations(roi, annotations, section_index, proteins=None, alpha=0.35, color_by="protein",
                        figsize=(8, 8), ax=None, show=True):
    """Show the original ROI section with the annotation polygons on top and a protein legend.

    annotations   RoiAnnotations from load_roi_annotations, or records already in ROI coordinates
                  (to_roi_coordinates). proteins=["GFAP", "CD68"] shows only those. `roi` is not modified.
    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Patch
    from matplotlib.path import Path

    recs = _records_for(annotations, section_index, proteins)
    sec = _section_rgb(roi, section_index)
    H, W = sec.shape[:2]
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(sec, extent=(0, W, H, 0), interpolation="nearest")
    counts = {}
    for rec in recs:
        rgb = tuple(c / 255 for c in _color(rec, color_by))
        for rings in _polys(rec["roi_geometry"]):
            verts, codes = [], []
            for ring in rings:
                r = np.asarray(ring, dtype=float)
                verts += list(r) + [r[0]]
                codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 1) + [Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(verts, codes), facecolor=rgb + (alpha,), edgecolor=rgb, linewidth=1))
        counts[rec["protein"]] = counts.get(rec["protein"], 0) + 1
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    title = f"Stack section {section_index}"
    if isinstance(annotations, RoiAnnotations) and annotations.atlas_sections:
        title += f" (Atlas SE {annotations.atlas_sections.get(section_index)})"
    ax.set_title(title + ("" if recs else " - no annotations in this ROI"), fontsize=11)
    if counts:
        ax.legend(handles=[Patch(facecolor=tuple(c / 255 for c in _hex_rgb(_COLORS[p])) + (alpha,),
                                 edgecolor=_COLORS[p], label=f"{p} - {STROKE_PROTEINS[p]} ({n})")
                           for p, n in counts.items()],
                  loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=8, frameon=False)
    if fig is not None:
        fig.tight_layout()
        if show:
            plt.show()
    return fig if fig is not None else ax.figure
