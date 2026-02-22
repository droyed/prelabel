# prelabel

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

> **TL;DR:** Push Ultralytic YOLO/SAM3 results directly onto Label Studio with one Python call and no middle layer of managing images, artifacts etc.

Toolkit for bridging YOLO/SAM model predictions with Label Studio annotation workflows.

Push YOLO inference results directly into Label Studio as pre-annotated tasks — bounding boxes, polygon masks, or brush masks — and export finished annotations in YOLO, COCO, or PNG format.

---

## Architecture

Two layers, separated by dependency weight:

```
┌─────────────────────────────────────────────────────────┐
│  ML Layer  (pip install prelabel[ml])                   │
│  ultralytics · opencv-python · numpy                    │
│  label-studio-converter                                 │
│                                                         │
│  extract_ls_predictions()  push_yolo_to_labelstudio()  │
└─────────────────────┬───────────────────────────────────┘
                      │ calls
┌─────────────────────▼───────────────────────────────────┐
│  Core Layer  (pip install prelabel)                     │
│  requests only  (~1 MB, no PyTorch)                     │
│                                                         │
│  LabelStudioClient — REST API wrapper                  │
└─────────────────────────────────────────────────────────┘
```

Use only the Core layer if you have pre-built annotations and just need to manage Label Studio projects via the REST API.

---

## Installation

**Core only** (Label Studio REST client, no ML dependencies):
```bash
pip install prelabel
```

**With ML support** (YOLO/SAM bridge, pulls in PyTorch ~3–5 GB):
```bash
pip install "prelabel[ml]"
```

**Install from GitHub repository
```bash
pip install git+https://github.com/droyed/prelabel.git
```

**With development tools**:
```bash
pip install "prelabel[ml,dev]"
```

**From source** (editable):
```bash
git clone <repo>
cd prelabel
pip install -e ".[ml,dev]"
```

---

## Quick Start

Start `label-studio` on a port. This port will be used in Python calls:

```console
label-studio start --port 8080
```

One function call to push a YOLO result into Label Studio:

```python
import os
from ultralytics import YOLO
from prelabel.label_studio_utils import push_yolo_to_labelstudio

model = YOLO("yolov8n-seg.pt")
results = model("assets/images/person.png")

push_yolo_to_labelstudio(
    yolo_result=results[0],
    img_path="assets/images/person.png",
    port=8080,
    api_key=os.getenv("LABELSTUDIO_TOKEN"),
    project_id=67,
    task_type="segmentation",   # or "detection"
    conf_threshold=0.5,
)

# Assets available at - http://localhost:8080/
```

---

## Usage

### Level 1 — One-liner: `push_yolo_to_labelstudio`

Best for quick inference-to-annotation pipelines. Creates a `LabelStudioClient` internally, extracts predictions from a YOLO result, and imports the image + predictions in one call.

```python
from prelabel.label_studio_utils import push_yolo_to_labelstudio

summary = push_yolo_to_labelstudio(
    yolo_result=results[0],
    img_path="image.png",
    port=8080,
    api_key="YOUR_API_KEY",
    project_id=42,
    task_type="detection",      # "detection" → RectangleLabels
    conf_threshold=0.4,
)
```

### Level 2 — Full workflow with `LabelStudioClient`

Use when you need full control over project creation, bulk import, or export.

```python
from prelabel import LabelStudioClient

ls = LabelStudioClient(8080, "YOUR_API_KEY")

# Create a project
project_id = ls.create_bbox_project("Smart City - Detection", ["Vehicle", "Pedestrian"])
# or: ls.create_polygon_project(...)
# or: ls.create_brush_project(...)

# Option A: Bulk import images from a local directory
#   Requires Label Studio started with LOCAL_FILES_SERVING_ENABLED=true
count = ls.import_local_images(project_id, "/path/to/images/")
print(f"Imported {count} images")

# Option B: Import a single image with pre-annotations
ls_predictions = [{
    "from_name": "label",
    "to_name": "image",
    "type": "rectanglelabels",
    "score": 0.88,
    "value": {
        "x": 25.0, "y": 25.0, "width": 50.0, "height": 40.0,
        "rotation": 0, "rectanglelabels": ["Vehicle"]
    }
}]
ls.import_preannotated_task(project_id, "image.png", ls_predictions)

# Check project status
ls.list_projects_summary()

# Export after annotation is complete
ls.export_bbox_yolo(project_id, output_path="dataset_yolo.zip")
# or: ls.export_polygon_coco(project_id, output_path="dataset_coco.zip")
# or: ls.export_brush_png(project_id, output_path="dataset_masks.zip")
```

### Level 3 — Custom flows with `extract_ls_predictions`

Extract Label Studio prediction dicts from a YOLO result without pushing them, for custom pipelines:

```python
from prelabel.label_studio_utils import extract_ls_predictions

ls_predictions = extract_ls_predictions(
    yolo_result=results[0],
    task_type="segmentation",   # "segmentation" → BrushLabels (RLE)
                                # "detection"    → RectangleLabels
    from_name="tag",            # must match your LS XML config
    to_name="image",
    conf_threshold=0.5,
)
# ls_predictions is a list of dicts ready for import_preannotated_task()
```

---

## Annotation Types

| Type            | LS Tag            | Export format      |
|-----------------|-------------------|--------------------|
| Bounding boxes  | `RectangleLabels` | YOLO `.txt` (ZIP)  |
| Polygons        | `PolygonLabels`   | COCO JSON (ZIP)    |
| Brush masks     | `BrushLabels`     | PNG rasterized (ZIP)|

---

## API Reference

### `LabelStudioClient(port, api_key)`

| Method | Returns | Description |
|--------|---------|-------------|
| `create_bbox_project(title, labels)` | `int` | Create bounding box project |
| `create_polygon_project(title, labels)` | `int` | Create polygon segmentation project |
| `create_brush_project(title, labels)` | `int` | Create brush mask project |
| `import_local_images(project_id, image_directory)` | `int` | Bulk-upload directory of images; returns count |
| `import_preannotated_task(project_id, image_path, ls_predictions, model_version)` | `dict` | Upload image + predictions atomically |
| `export_bbox_yolo(project_id, output_path)` | `str` | Export YOLO format ZIP |
| `export_polygon_coco(project_id, output_path)` | `str` | Export COCO JSON ZIP |
| `export_brush_png(project_id, output_path)` | `str` | Export PNG masks ZIP |
| `list_projects_summary()` | `None` | Print tabular status of all projects |
| `cleanup_empty_projects()` | `list[str]` | Delete zero-task projects; returns deleted titles |

### Module-level functions

| Function | Description |
|----------|-------------|
| `extract_ls_predictions(yolo_result, task_type, from_name, to_name, conf_threshold)` | Convert YOLO result → LS prediction dicts |
| `push_yolo_to_labelstudio(yolo_result, img_path, port, api_key, project_id, task_type, conf_threshold)` | One-call YOLO → Label Studio push |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LABELSTUDIO_PORT` | `8080` | Label Studio server port |
| `LABELSTUDIO_TOKEN` | — | API key (from Account & Settings in Label Studio) |
| `LS_TEST_PORT` | — | Override port for tests |
| `LS_TEST_API_KEY` | — | Override API key for tests |

---

## Running Everything

To run both demos and the test suite in one shot:

```bash
./run_all.sh
```

This executes `demos/basic_usage.py`, `demos/demo_LabelStudioClient.py`, and `tests/test_LabelStudioClient.py` in order, stopping on the first failure.

> **Note:** `basic_usage.py` requires a real image at `assets/images_YOLO/person_207.png` and `LABELSTUDIO_TOKEN` set. The other scripts use dummy images.

## Running Tests

Tests require a live Label Studio instance. Set credentials via environment variables:

```bash
export LS_TEST_PORT=8080
export LS_TEST_API_KEY="your-api-key"

pytest
# or with coverage:
pytest --cov
```

---

## Attributions

Test images located at `assets/images_YOLO` are from the [COCO val2017 dataset](https://cocodataset.org/) (Lin et al., 2015), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.