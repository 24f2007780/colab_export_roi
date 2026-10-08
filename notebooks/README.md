# Tutorial & Analysis Notebooks

Back to [Top](../README.md) | [Documentation](../docs/README.md) | [Data](../data/README.md)

This directory contains Jupyter notebooks demonstrating programmatic access, multi-modal alignment, digital pathology annotations, and MRI lesion analysis for post-mortem stroke research.

## List of Notebooks

| Aspect / Domain | Notebook | Description | Key Modules / Dependencies |
| :--- | :--- | :--- | :--- |
| **Multi-Modal 3D Visualization** | [`multi_modal_3d_visualization_colab.ipynb`](multi_modal_3d_visualization_colab.ipynb) | Comprehensive Colab/local interactive workflow demonstrating cross-modal alignment between histology ROI stacks, digital pathology annotations, and reference MRI spaces. | `roi_utils`, `annotations_utils`, Matplotlib |
| **Stroke Annotations** | [`stroke_annotations.ipynb`](stroke_annotations.ipynb) | Digital pathology annotation querying, multi-protein GeoJSON retrieval (CD68, GFAP, Fib, CD34, HIF1a), ROI coordinate transforms, filtering, and tissue overlap validation. | `annotations_utils`, OpenAtlas API |
| **Stroke MRI Analysis** | [`stroke_mri.ipynb`](stroke_mri.ipynb) | Core neuroimaging exploration of stroke lesion markers, DWI hyperintensity, ADC hypointensity, and intensity distributions. | `roi_utils`, `annotations_utils`, `mri_utils` |
| **MRI Testing & Validation** | [`MRI_testing.ipynb`](MRI_testing.ipynb) | Comprehensive verification suite for NIfTI geometry, voxel-to-world transforms, brain masking, paired diffusion analysis (iv11 vs iv16), and atlas burden lookups. | `mri_utils`, `nibabel` |
| **MRI Library Exploration** | [`lib-mri.ipynb`](lib-mri.ipynb) | Comparative evaluation using NiBabel and Nilearn for spatial resampling, registration to ex-vivo space, and crosshair mapping. | `nibabel`, `nilearn`, `mri_utils` |

## Execution Guidelines

1. **Path Resolution**: The notebooks automatically add the repository root to `sys.path` (`sys.path.append('..')`) to import `roi_utils`, `annotations_utils`, and `mri_utils`.
2. **Data Directory**: Imaging volumes and metadata are organized in [`../data/`](../data/README.md). Functions in `mri_utils` automatically resolve data paths across `.` and `data/`.
3. **Environment**: Ensure dependencies are installed via `pip install -r requirements.txt`.
