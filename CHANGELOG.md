# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2026-02-23

### Added
- `demos/demo_det.py`: New unified demo script supporting both YOLO and SAM3 modes via CLI argument (`python demo_det.py yolo|sam3`), replacing `basic_usage.py`.
- `label_studio_utils.generate_label_config()`: Generates a Label Studio XML config string from a list of label dicts.
- `LabelStudioClient.create_cv_project_BrushLabels()`: Creates a new CV project in Label Studio with BrushLabels configuration.
- `LabelStudioClient.get_projects_summary()`: Returns a structured list of dicts summarising all Label Studio projects (ID, title, classes, tasks, progress, etc.).
- `LabelStudioClient.setup_project_interactive()`: Interactive CLI prompt to select an existing project or create a new one.
- `label_studio_utils.rgb_to_hex()`: Utility to convert an RGB list/tuple to a hex colour string.
- `label_studio_utils.generate_yolo_labels()`: Generates label name/colour dicts from a YOLO Results object for use with Label Studio.
- `check_label_studio_running()`: Added `raise_on_error` parameter; `LabelStudioClient.__init__` now calls it with `raise_on_error=True` to fail fast on startup.

### Changed
- `run_all.sh`: Updated `run()` helper to forward extra CLI arguments to scripts; updated run list to invoke `demo_det.py` with `yolo` and `sam3` modes.
- `extract_ls_predictions()`: Removed `.capitalize()` from class name lookup to preserve original casing.

### Removed
- `demos/basic_usage.py`: Replaced by the more capable `demos/demo_det.py`.
