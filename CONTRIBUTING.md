# Contributing to prelabel

Thanks for your interest in contributing! `prelabel` is a small, focused toolkit and contributions
that keep it simple and well-tested are especially welcome.

---

## Getting Set Up

```bash
git clone <repo-url> && cd <repo-dir>
pip install -e .[dev]   # editable install + dev/test tools
```

Set your Label Studio token before running any live tests:

```bash
export LABELSTUDIO_TOKEN="your-token-here"
```

---

## Running Tests

```bash
pytest
```

Tests that hit a live Label Studio instance are skipped automatically when
`LABELSTUDIO_TOKEN` is not set or the server is unreachable.

---

## We Are Looking for Contributors: New Labeling Types

**This is the highest-impact area for new contributors.**

`prelabel` currently supports three Label Studio annotation types:

| Task Type | Label Studio tag | Status |
|-----------|-----------------|--------|
| Bounding Box | `RectangleLabels` | Supported |
| Brush / Semantic Seg | `BrushLabels` (RLE) | Supported |
| Instance Segmentation | `PolygonLabels` | Supported (manual import only) |

We would love help adding support for more Label Studio label types, for example:

- **`KeypointLabels`** — pose estimation output from YOLO-Pose
- **`EllipseLabels`** — ellipse annotations
- **`TimeSeriesLabels`** — time-series data labeling
- **`VideoRectangle`** — bounding boxes on video frames
- **`AudioClassification`** / **`AudioRegion`** — audio annotations

To add a new type, follow the pattern in `src/prelabel/label_studio_utils.py`:

1. Add an `extract_ls_<type>_predictions(results, label_config)` function that converts model
   output into a list of Label Studio prediction dicts.
2. Register the new `task_type` key in the `extract_ls_predictions` dispatcher.
3. Add a corresponding label config template string (see existing examples).
4. Write at least one unit test in `tests/`.

Open an issue first to discuss the approach before sending a PR — it saves everyone time.

---

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Keep commits focused; one logical change per commit.
3. Run `pytest` and confirm all tests pass before opening the PR.
4. Open a PR against `main` with a clear description of what changed and why.

For bug reports or feature requests, please open a GitHub issue.

---

## Code Style

- Follow existing conventions in the file you are editing.
- `black` and `isort` are included in the `[dev]` extras — run them before committing:

  ```bash
  black src/ tests/
  isort src/ tests/
  ```

- Prefer clarity over cleverness; this codebase is meant to be easy to read and extend.
