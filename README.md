# prelabel ![version](https://img.shields.io/badge/version-1.0.0-blue)

Python utility toolkit for driving Label Studio directly from code — pre-labeling, batch import, project management, and debugging.

---

## Overview

`prelabel` bridges YOLO and SAM model predictions with Label Studio annotation workflows. It
converts raw model output (bounding boxes, segmentation masks, brush-encoded RLE) into the Label
Studio prediction format and uploads everything — images and predictions together — via the
Label Studio REST API. The result is a fully pre-annotated project that human reviewers can
immediately correct and approve.

Beyond the standard pre-labeling pipeline, `prelabel` is also useful as a Python debugging and
inspection library. Every operation that you would normally perform in the Label Studio web UI
— creating projects, listing their status, cleaning up empty ones, or wiping everything for a
fresh start — is available as a direct Python call. This makes it straightforward to script
housekeeping tasks, verify connectivity, and introspect the state of your annotation server
without opening a browser.

---

## Architecture

```
Core layer (requests only)
  └── LabelStudioClient          ← project CRUD, import, export, status

ML layer (+ultralytics, numpy, label-studio-converter)
  ├── extract_ls_predictions         ← unified dispatcher
  ├── extract_ls_bbox_predictions    ← YOLO boxes → rectanglelabels
  ├── extract_ls_segmentation_predictions  ← YOLO masks → brushlabels (RLE)
  ├── generate_yolo_labels_from_classnames ← class names → label dicts with colors
  └── yolo_to_labelstudio      ← high-level YOLO→LS adapter
```

## Installation

```bash
git clone <repo-url> && cd <repo-dir>
pip install -e .[dev]   # editable + dev tools
pip install -e .[all]   # editable + all test/dev packages
pip install .              # core only (no ultralytics/numpy)
```

## Setup

```bash
# Start Label Studio (add LOCAL_FILES_SERVING_ENABLED=true for bulk local image import)
label-studio start --port 8080

# Set your API token (Account & Settings → Access Token in the UI)
export LABELSTUDIO_TOKEN="your-token-here"
```

## Quick Start

```python
from ultralytics import YOLO
from prelabel import yolo_to_labelstudio

model = YOLO("yolov8n-seg.pt")
results = model.predict(source=["image1.jpg", "image2.jpg"])

yolo_to_labelstudio(results, task_type="segmentation", project_title="SegProj")
```

## Workflows

| Workflow | Description |
|----------|-------------|
| **1. YOLO Bounding Boxes** | Create a bbox project, import images with pre-computed predictions, export in YOLO format. |
| **2. Bulk Data Ingestion** | Create a polygon project, import a whole image directory, export in COCO format. |
| **3. Full YOLO Segmentation** | Run YOLO inference and create both a segmentation and a bbox project in one script. |
| **4. SAM3 Semantic Segmentation** | Use SAM3 text-guided segmentation and push results as a brush-label project. |
| **5. Debugging & Inspection** | Script housekeeping tasks (list, clean up, delete projects) without the web UI. |

## API Reference

See [API_Reference.md](API_Reference.md) for full method signatures, parameter types,
return values, exceptions raised, and HTTP endpoints used.

## Supported Annotation Types

| Task Type | Label Studio tag | Prediction `type` | Export format |
|-----------|-----------------|-------------------|---------------|
| Bounding Box | `RectangleLabels` | `rectanglelabels` | YOLO (`.zip`) |
| Instance Segmentation | `PolygonLabels` | *(raw polygons, manual)* | COCO JSON (`.zip`) |
| Brush / Semantic Seg | `BrushLabels` | `brushlabels` (RLE) | PNG masks (`.zip`) |

## Project Structure

```
├── pyproject.toml              # Build config, dependencies, optional extras
├── src/prelabel/
│   ├── __init__.py             # Public API: LabelStudioClient, yolo_to_labelstudio
│   ├── label_studio_utils.py   # LabelStudioClient, prediction extractors, utilities
│   └── adapters.py             # High-level YOLO→Label Studio adapter
├── demos/
│   ├── demo_LabelStudioClient.py   # Workflows 1 & 2 (client-level)
│   └── demo_get_started.py         # Workflows 3 & 4 (YOLO + SAM3)
└── tests/
    ├── test_LabelStudioClient.py
    └── test_adapters.py
```

## Running the Demos and test scripts

```bash
# Ensure LABELSTUDIO_TOKEN is set and Label Studio is running on port 8080
make test demo
```

## 🎬 Demo

- [▶ How to use the codebase](https://github.com/droyed/prelabel/releases/download/v1.0.0/HowToUse.mp4)
- [▶ Use case walkthrough](https://github.com/droyed/prelabel/releases/download/v1.0.0/UseCase1.mp4)

---

## Attributions

Test images located at `assets/images_YOLO` are from the [COCO val2017 dataset](https://cocodataset.org/) (Lin et al., 2015), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup
instructions, how to add support for new labeling types, and the pull request
process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.