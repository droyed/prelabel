# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **`.gitignore`**: Excluded `bpe_simple_vocab_16e6.txt.gz` (BPE vocabulary asset) from version control.

## [1.0.0] - 2026-03-01

### Added
- **`src/prelabel/adapters.py`**: New adapter module with `yolo_to_labelstudio()` — high-level function that accepts YOLO/SAM Results, auto-generates label colors, creates or reuses a Label Studio project, and batch-imports predictions. Supports `"segmentation"` and `"bbox"` task types.
- **`LabelStudioClient.create_cv_project_generic()`**: Creates a project with a configurable control type (`BrushLabels`, `RectangleLabels`, `PolygonLabels`, etc.).
- **`LabelStudioClient.create_brush_project()`**: Convenience wrapper around `create_cv_project_generic` for BrushLabels projects.
- **`LabelStudioClient.get_projects_summary()`**: Returns a structured list of dicts with project metadata (ID, title, classes, tasks, progress, etc.).
- **`label_studio_utils.generate_yolo_labels_from_classnames()`**: Generates label dicts with visually distinct colors from YOLO class names (list or `{int: str}` dict), using imtools golden-ratio color generation.
- **`label_studio_utils.extract_ls_bbox_predictions()`**: Extracts YOLO bounding-box predictions as Label Studio `rectanglelabels` region dicts.
- **`label_studio_utils.extract_ls_segmentation_predictions()`**: Extracts YOLO mask predictions as Label Studio `brushlabels` region dicts (with RLE encoding via `label-studio-converter`).
- **`label_studio_utils.extract_ls_predictions()`**: Unified dispatcher that delegates to the bbox or segmentation extractor; raises `ValueError` for unknown task types.
- **`label_studio_utils._validate_and_filter()`**: Internal helper to check boxes/masks existence and apply a confidence threshold.
- **`label_studio_utils._make_base_region()`**: Internal helper to build the shared base dict for any region type.
- **`API_Reference.md`**: Comprehensive API reference documenting all public methods, parameters, return types, and prediction format schemas.
- **`Makefile`**: Build automation with `test`, `demo`, `clean`, and `help` targets; replaces `run_all.sh`.
- **`demos/demo_get_started.py`**: New end-to-end getting-started demo covering YOLO segmentation, bounding-box, and SAM3 semantic segmentation workflows using the new `yolo_to_labelstudio` adapter.
- **`tests/test_adapters.py`**: Test suite for the adapter layer, covering new-project and existing-project creation for both `segmentation` and `bbox` modes, confidence-threshold filtering, invalid task-type errors, and results-without-masks skipping.
- **`tests/test_LabelStudioClient.py`**: Added `test_get_projects_summary()` test; added brush project type to `test_project_creation()`; improved PNG export test to handle 500 response gracefully.
- **`pyproject.toml`**: Added optional dependency groups `ml` (ultralytics, numpy, label-studio-converter, imtools) and `dev` (pytest, pytest-cov); added wheel exclude rules for assets, models, and test data; added hatchling `allow-direct-references` for imtools GitHub source; added pytest and coverage config.

### Changed
- **Version**: Bumped to `1.0.0` in `pyproject.toml` and `src/prelabel/__init__.py`.
- **`src/prelabel/__init__.py`**: Public API now explicitly exports `LabelStudioClient` and `yolo_to_labelstudio`; sets `__version__ = "1.0.0"`.
- **`LabelStudioClient.__init__`**: Now calls `check_label_studio_running(port, raise_on_error=True)` on startup — fails fast with `ConnectionError` if Label Studio is not reachable.
- **`check_label_studio_running()`**: Added `raise_on_error` parameter; when `True`, raises `ConnectionError` instead of returning `False`.
- **`demos/demo_LabelStudioClient.py`**: Refactored to demonstrate two canonical workflows — AI-assisted bounding-box annotation (with pre-annotations) and bulk data ingestion/maintenance (with COCO export and project cleanup).
- **`README.md`**: Complete rewrite for v1.0.0 — added version badge, clearer architecture overview, installation extras syntax, setup instructions, expanded Quick Start, workflows table, supported annotation types table, pointer to `API_Reference.md`, updated project structure, and attributions section.
- **`label_studio_utils.generate_label_config()`**: Now accepts a `label_type` parameter (default `"BrushLabels"`) to generate the XML config for any Label Studio control type.
- **Core dependency**: `pyproject.toml` now lists only `requests>=2.28.0` as a required dependency; all ML libraries moved to optional `[ml]` group.

### Removed
- **`demos/demo_det.py`**: Replaced by `demos/demo_get_started.py`.
- **`run_all.sh`**: Replaced by `Makefile`.
