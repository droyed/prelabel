# prelabel API Reference

---

## `LabelStudioClient`

```python
from prelabel import LabelStudioClient

client = LabelStudioClient(port=8080, api_key="your-token")
```

**Constructor behavior:** Calls `check_label_studio_running()` immediately. Raises `ConnectionError` if Label Studio is not reachable on the given port. Sets up a persistent `requests.Session` with the `Authorization: Token <api_key>` header.

---

### Project Management

| Method | Signature | Returns | HTTP |
|--------|-----------|---------|------|
| `create_bbox_project` | `(title: str, labels: list[str])` | `int` | `POST /api/projects/` |
| `create_polygon_project` | `(title: str, labels: list[str])` | `int` | `POST /api/projects/` |
| `create_brush_project` | `(title: str, labels: list[str])` | `int` | `POST /api/projects/` |
| `create_cv_project_generic` | `(title: str, labels: list[dict], label_type: str)` | `int` | `POST /api/projects/` |
| `project_exists` | `(project_id: int, raise_on_missing: bool = True)` | `bool` | `GET /api/projects/{id}/` |
| `list_projects_summary` | `()` | `None` | `GET /api/projects/` |
| `get_projects_summary` | `()` | `list[dict]` | `GET /api/projects/` |
| `cleanup_empty_projects` | `()` | `list[str]` | `GET /api/projects/` + `DELETE /api/projects/{id}/` |
| `delete_all_projects` | `()` | `int` | `GET /api/projects/` + `DELETE /api/projects/{id}/` |

**`create_bbox_project` / `create_polygon_project` / `create_brush_project`**

- `labels`: plain list of class name strings, e.g. `["Cat", "Dog"]`
- Internally calls `_create_project_from_tag` with `RectangleLabels`, `PolygonLabels`, or `BrushLabels` respectively
- The generated label config uses `name="label"` (not `"tag"`) — relevant for `from_name` when building predictions
- Returns the new project's integer ID

**`create_cv_project_generic`**

- `labels`: list of dicts, each with `"name"` and `"color"` keys:
  ```python
  [{"name": "Person", "color": "#FFA39E"}, {"name": "Car", "color": "#D4380D"}]
  ```
- `label_type`: must be `"BrushLabels"` or `"RectangleLabels"`
- The generated label config uses `name="tag"` — used by `yolo_to_labelstudio`
- Returns the new project's integer ID

**`project_exists`**

- Returns `True` if `GET /api/projects/{project_id}/` returns HTTP 200
- Returns `False` (or raises) if HTTP 404
- Raises `ValueError` when `raise_on_missing=True` and project is not found
- Calls `response.raise_for_status()` for any other unexpected status (e.g. 401 Unauthorized)

**`list_projects_summary`**

- Prints a formatted 90-char-wide table to stdout with columns: ID, Title (truncated to 22 chars), Classes, Tasks, Annotated, Progress, Annots, Created Date
- Returns `None`

**`get_projects_summary`**

- Returns the same data as `list_projects_summary` as a `list[dict]`
- Each dict has keys: `"ID"`, `"Title"`, `"Classes"`, `"Tasks"`, `"Annotated"`, `"Progress"`, `"Annots"`, `"Created Date"`
- Returns `[]` if no projects exist

**`cleanup_empty_projects`**

- Deletes all projects with `task_number == 0`
- Returns a `list[str]` of deleted project titles
- Prints per-deletion status and a final summary count

**`delete_all_projects`**

- Deletes every project on the instance (HTTP 204 = success)
- Returns `int` count of successfully deleted projects

---

### Import

**`import_preannotated_task`**

```python
import_preannotated_task(
    project_id: int,
    image_path: str,
    ls_predictions: list[dict],
    model_version: str = "yolo-model"
) -> dict
```

- Validates project exists (raises `ValueError` via `project_exists`)
- Reads the image file and encodes it as a Base64 Data URI
- Raises `FileNotFoundError` if `image_path` does not exist
- Posts a single-task payload to `POST /api/projects/{project_id}/import`
- Raises `ValueError` if the API returns `task_count == 0`
- Returns the raw import response dict

**`import_preannotated_tasks_batch`**

```python
import_preannotated_tasks_batch(
    project_id: int,
    batch_data: list[dict],
    model_version: str = "yolo-model",
    batch_size: int = 25
) -> int
```

- `batch_data` schema:
  ```python
  [
      {
          "image_path": str,   # Absolute or relative path to the image file
          "predictions": list  # List of Label Studio region dicts
      },
      ...
  ]
  ```
- Validates project exists (raises `ValueError` via `project_exists`)
- Processes `batch_data` in chunks of `batch_size` to avoid large JSON payloads
- Skips missing image files with a warning (`⚠️ Warning: Image not found at ...`), continues with remaining images
- Posts each chunk to `POST /api/projects/{project_id}/import`
- Raises `ValueError` if total tasks created across all batches is 0
- Returns `int` total number of tasks successfully created

**`import_local_images`**

```python
import_local_images(project_id: int, image_directory: str) -> int
```

- Scans `image_directory` for files with extensions `.jpg`, `.jpeg`, `.png`, `.bmp` (case-insensitive)
- Uploads each image as multipart/form-data to `POST /api/projects/{project_id}/import`
- **Requires** Label Studio to be started with `LOCAL_FILES_SERVING_ENABLED=true`
- Returns `int` count of successfully uploaded images (HTTP 200 or 201)

---

### Export

| Method | Signature | Returns | HTTP |
|--------|-----------|---------|------|
| `export_bbox_yolo` | `(project_id: int, output_path: str = "yolo_bboxes.zip")` | `str` | `GET /api/projects/{id}/export?exportType=YOLO` |
| `export_polygon_coco` | `(project_id: int, output_path: str = "coco_polygons.zip")` | `str` | `GET /api/projects/{id}/export?exportType=COCO` |
| `export_brush_png` | `(project_id: int, output_path: str = "png_masks.zip")` | `str` | `GET /api/projects/{id}/export?exportType=PNG` |
| `export_annotations` | `(project_id: int, export_type: str, output_path: str)` | `str` | `GET /api/projects/{id}/export?exportType={type}` |

**`export_annotations`** (core method)

- `export_type` values: `"YOLO"`, `"COCO"`, `"PNG"`
- Writes the response body (a ZIP file) to `output_path`
- Returns `output_path` (the string passed in)
- The three wrapper methods (`export_bbox_yolo`, `export_polygon_coco`, `export_brush_png`) call this with fixed export types

---

## `yolo_to_labelstudio`

```python
from prelabel import yolo_to_labelstudio

proj_id = yolo_to_labelstudio(
    results,                                          # list[ultralytics.engine.results.Results]
    task_type="segmentation",                         # str
    projectID=None,                                   # int | None
    port=8080,                                        # int
    labelstudio_token_name="LABELSTUDIO_TOKEN",       # str
    project_title="Demo",                             # str
    model_version="yolov8n-model-v1",                 # str
    batch_size=25,                                    # int
    conf_threshold=0.0                                # float
) -> int
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `list` | — | List of Ultralytics `Results` objects from `model.predict(...)` |
| `task_type` | `str` | `"segmentation"` | `"segmentation"` (BrushLabels) or `"bbox"` (RectangleLabels) |
| `projectID` | `int \| None` | `None` | `None` = create new project; `int` = append to existing project |
| `port` | `int` | `8080` | Label Studio server port |
| `labelstudio_token_name` | `str` | `"LABELSTUDIO_TOKEN"` | Environment variable name holding the API token |
| `project_title` | `str` | `"Demo"` | Title for the new project (ignored when `projectID` is provided) |
| `model_version` | `str` | `"yolov8n-model-v1"` | Model version string attached to predictions |
| `batch_size` | `int` | `25` | Tasks per API call; reduce if hitting payload size limits |
| `conf_threshold` | `float` | `0.0` | Drop predictions with confidence below this value |

**Returns:** `int` — the project ID (newly created or the provided `projectID`)

**Raises:** `ValueError` if `task_type` is not `"segmentation"` or `"bbox"`

**Task-type mapping:**

| `task_type` | Label Studio tag | YOLO attribute checked | `from_name` in predictions |
|-------------|-----------------|----------------------|---------------------------|
| `"segmentation"` | `BrushLabels` | `result.masks` | `"tag"` |
| `"bbox"` | `RectangleLabels` | `result.boxes` | `"tag"` |

Results where the relevant attribute is `None` are silently skipped.

---

## Prediction Extractors (ML layer)

```python
from prelabel.label_studio_utils import (
    extract_ls_predictions,
    extract_ls_bbox_predictions,
    extract_ls_segmentation_predictions,
)
```

**`extract_ls_predictions`** — unified dispatcher

```python
extract_ls_predictions(
    yolo_result,
    task_type: str = "bbox",
    **kwargs
) -> tuple[list[dict], set[str]]
```

- Dispatches to `extract_ls_bbox_predictions` or `extract_ls_segmentation_predictions` based on `task_type`
- `task_type` values: `"bbox"`, `"segmentation"`
- Raises `ValueError` for unknown `task_type`
- Passes `**kwargs` directly to the underlying extractor

**`extract_ls_bbox_predictions`**

```python
extract_ls_bbox_predictions(
    yolo_result,
    from_name: str = "tag",
    to_name: str = "image",
    conf_threshold: float = 0.0
) -> tuple[list[dict], set[str]]
```

- Returns `(list[dict], set[str])` — list of rectanglelabels region dicts + set of detected class names
- Returns `([], set())` if no boxes or all boxes below `conf_threshold`
- Coordinates are in Label Studio's percentage system (0–100%), derived from normalized YOLO `xyxyn` boxes

**`extract_ls_segmentation_predictions`**

```python
extract_ls_segmentation_predictions(
    yolo_result,
    from_name: str = "tag",
    to_name: str = "image",
    conf_threshold: float = 0.0
) -> tuple[list[dict], set[str]]
```

- Returns `(list[dict], set[str])` — list of brushlabels region dicts + set of detected class names
- Returns `([], set())` if no boxes or all boxes below `conf_threshold`
- Masks are scaled to original image size and encoded as RLE using `label_studio_converter.brush.mask2rle`

---

## Utility Functions

```python
from prelabel.label_studio_utils import (
    generate_yolo_labels_from_classnames,
    generate_label_config,
    check_label_studio_running,
    rgb_to_hex,
)
```

**`generate_yolo_labels_from_classnames`**

```python
generate_yolo_labels_from_classnames(
    class_names: list[str] | dict[int, str]
) -> list[dict]
```

- Accepts either a list of class name strings or a dict mapping int indices to strings
- Generates visually distinct colors using a golden-ratio color generation algorithm (`imtools.viz.colors.generate_colors`)
- Returns `list[{"name": str, "color": str}]` — hex color strings like `"#ff0000"`

**`generate_label_config`**

```python
generate_label_config(
    labels: list[dict],
    label_type: str = "BrushLabels"
) -> str
```

- `labels`: list of `{"name": str, "color": str}` dicts
- `label_type`: any valid Label Studio control tag — e.g. `"BrushLabels"`, `"RectangleLabels"`, `"PolygonLabels"`
- Returns a stripped XML string with `<View>`, `<Image name="image">`, and the label tag using `name="tag"`

**`check_label_studio_running`**

```python
check_label_studio_running(
    port: int,
    timeout: int = 5,
    raise_on_error: bool = False
) -> bool
```

- Makes `GET http://localhost:{port}/health`
- Returns `True` if HTTP 200
- If not running: prints an error message and returns `False` (default), or raises `ConnectionError` if `raise_on_error=True`
- Timeout errors and connection errors are both caught

**`rgb_to_hex`**

```python
rgb_to_hex(rgb: list[int] | tuple[int, int, int]) -> str
```

- Converts `[r, g, b]` or `(r, g, b)` to `"#rrggbb"`
- Clamps each channel to the range 0–255 before conversion
- Raises `ValueError` if `len(rgb) != 3`

---

## Prediction Format Reference

### `rectanglelabels` region dict

```python
{
    "id": "a1b2c3d4",           # str — 8-char UUID fragment
    "origin": "manual",
    "type": "rectanglelabels",
    "from_name": "label",       # str — must match label config name= attribute
                                #   "label" for bbox/polygon projects
                                #   "tag" for brush projects and yolo_to_labelstudio
    "to_name": "image",         # str — matches <Image name="image"/>
    "image_rotation": 0,
    "original_width": 1280,     # int — source image width in pixels
    "original_height": 720,     # int — source image height in pixels
    "score": 0.87,              # float — detection confidence
    "meta": {"text": ["Conf: 87.00%"]},
    "value": {
        "x": 12.5,              # float — left edge as % of image width (0–100)
        "y": 20.0,              # float — top edge as % of image height (0–100)
        "width": 35.0,          # float — box width as % of image width
        "height": 25.0,         # float — box height as % of image height
        "rotation": 0,
        "rectanglelabels": ["Cat"],  # list[str] — detected class name
    },
}
```

### `brushlabels` region dict

```python
{
    "id": "e5f6g7h8",           # str — 8-char UUID fragment
    "origin": "manual",
    "type": "brushlabels",
    "from_name": "tag",         # str — matches label config name="tag"
    "to_name": "image",
    "image_rotation": 0,
    "original_width": 1280,
    "original_height": 720,
    "score": 0.91,
    "meta": {"text": ["Conf: 91.00%"]},
    "value": {
        "format": "rle",        # str — always "rle" for brush predictions
        "rle": [...],           # list[int] — run-length encoded mask (label-studio-converter format)
        "brushlabels": ["Dog"], # list[str] — detected class name
    },
}
```

### `from_name` matching

The `from_name` field in each region dict must exactly match the `name` attribute of the corresponding control tag in the project's label config XML:

| Project created by | Label config `name=` | `from_name` to use |
|-------------------|---------------------|-------------------|
| `create_bbox_project` | `"label"` | `"label"` |
| `create_polygon_project` | `"label"` | `"label"` |
| `create_brush_project` | `"label"` | `"label"` |
| `create_cv_project_generic` | `"tag"` | `"tag"` |
| `yolo_to_labelstudio` | `"tag"` (via `create_cv_project_generic`) | `"tag"` (extractor default) |
