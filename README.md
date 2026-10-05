# ROI Utils

A tiny, single-file helper for working with the NumPy ROI stacks exported by our ROI Extractor.
You don't need to know anything about the backend: load a stack, look at it, clean it up a bit, and hand it to your ML code.

Only **NumPy** is required. matplotlib (previews), requests (Google Drive) and PyTorch (`to_torch`) are loaded only when you use those functions, and Colab already has them.

## What's an ROI stack?

A NumPy array where each section is stacked along the first axis:

| Shape | Meaning |
|---|---|
| `(Z, H, W)` | grayscale sections |
| `(Z, H, W, C)` | colour sections (C = 3 for RGB) |

## Get started (Google Colab)

```python
!wget -q https://raw.githubusercontent.com/24f2007780/colab_export_roi/main/roi_utils.py
from roi_utils import *

roi = load_and_preview_roi("<Google Drive file ID or share link>")
```

The Drive file must be shared as "Anyone with the link". You'll see the file name, shape, size, parsed details (centre X/Y, section, stain, resolution) and a preview of about 10 evenly spaced sections.

Already have a `.npy` file? `roi = load_roi("my_roi.npy")`

## Look at your data

```python
roi_info(roi)                                  # shape, dtype, size, min/max/mean/std

preview_sections(roi, start=100, end=150)      # sections 100 to 149 (end is not included)
preview_sections(roi, start=100, end=150, step=10)
preview_sections(roi, sections=[10, 25, 80])   # exactly these sections
```

Long ranges are thinned to 30 images so the plot stays quick.

## Prepare it for ML

```python
sub  = get_sections(roi, 100, 150)             # a sub-stack (no copy for a range)
gray = to_grayscale(roi)                       # RGB -> one channel
box  = center_crop(gray, 256)                  # or crop_roi(gray, y=(0, 200), x=(50, 250))
small = downsample_roi(box, 2)                 # half the height and width
x    = normalize_roi(small, clip_percentiles=(1, 99))   # float32 in [0, 1]
t    = to_torch(x)                             # tensor shaped (N, C, H, W)
```

| Function | What it does |
|---|---|
| `load_and_preview_roi(drive_id)` | Download from Google Drive, print details, preview |
| `load_roi(path, mmap=False)` | Load a local `.npy` (use `mmap=True` for stacks bigger than RAM) |
| `preview_sections(roi, ...)` | Show a range or a chosen list of sections |
| `get_sections(roi, ...)` | Pull out a sub-stack |
| `roi_info(roi)` | Quick summary and statistics |
| `normalize_roi(roi, method="minmax" or "zscore")` | Rescale intensities; never changes your original |
| `crop_roi` / `center_crop` | Crop every section (no copy) |
| `downsample_roi(roi, factor)` | Shrink by averaging blocks, or `method="stride"` for speed |
| `to_grayscale(roi)` | RGB to grayscale |
| `to_torch(roi)` | NumPy to PyTorch tensor |

## Good to know

- Stacks like ours are mostly white background, so use `clip_percentiles=(1, 99)` when normalising to keep tissue contrast.
- `normalize_roi` makes one float32 copy; `inplace=True` skips it for writable float arrays.
- Files are loaded with `allow_pickle=False` for safety. Plain image arrays are unaffected.
- A memory-mapped stack (`mmap=True`) is read-only; functions that need to change data copy it for you.
