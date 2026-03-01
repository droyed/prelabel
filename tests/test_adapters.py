import os
import glob
from pathlib import Path
from ultralytics import YOLO

from prelabel import LabelStudioClient, yolo_to_labelstudio

LS_PORT = int(os.getenv("LS_TEST_PORT", 8080))
API_KEY = os.getenv("LS_TEST_API_KEY", os.getenv("LABELSTUDIO_TOKEN"))

_THIS_DIR  = Path(__file__).parent
_PROJ_ROOT = _THIS_DIR.parent
MODEL_PATH = _PROJ_ROOT / "yolov8n-seg.pt"
IMAGE_DIR  = _PROJ_ROOT / "assets/images_YOLO"


def test_seg_new_project(client, results_seg):
    print("Testing seg new project creation via adapter...")
    proj_id = yolo_to_labelstudio(
        results_seg,
        task_type="segmentation",
        projectID=None,
        port=LS_PORT,
        project_title="Test-Adapter-Seg-New",
        model_version="test-seg-model-v1",
    )
    assert isinstance(proj_id, int) and proj_id > 0
    assert client.project_exists(proj_id, raise_on_missing=False), \
        f"Project {proj_id} returned but not found on server"
    print(f"  test_seg_new_project: PASS (project ID {proj_id})")
    return proj_id


def test_bbox_new_project(client, results_bbox):
    print("Testing bbox new project creation via adapter...")
    proj_id = yolo_to_labelstudio(
        results_bbox,
        task_type="bbox",
        projectID=None,
        port=LS_PORT,
        project_title="Test-Adapter-Bbox-New",
        model_version="test-bbox-model-v1",
    )
    assert isinstance(proj_id, int) and proj_id > 0
    print(f"  test_bbox_new_project: PASS (project ID {proj_id})")
    return proj_id


def test_seg_existing_project(client, results_seg):
    print("Testing seg import into existing project...")
    existing_id = client.create_brush_project("Test-Adapter-Seg-Existing", ["person", "car"])
    assert existing_id > 0
    returned_id = yolo_to_labelstudio(
        results_seg,
        task_type="segmentation",
        projectID=existing_id,
        port=LS_PORT,
    )
    assert returned_id == existing_id, f"Expected {existing_id}, got {returned_id}"
    print(f"  test_seg_existing_project: PASS (project ID {existing_id})")
    return existing_id


def test_bbox_existing_project(client, results_bbox):
    print("Testing bbox import into existing project...")
    existing_id = client.create_bbox_project("Test-Adapter-Bbox-Existing", ["person", "car"])
    assert existing_id > 0
    returned_id = yolo_to_labelstudio(
        results_bbox,
        task_type="bbox",
        projectID=existing_id,
        port=LS_PORT,
    )
    assert returned_id == existing_id, f"Expected {existing_id}, got {returned_id}"
    print(f"  test_bbox_existing_project: PASS (project ID {existing_id})")
    return existing_id


def test_conf_threshold_filtering(client, results_seg):
    """
    conf_threshold=0.999 filters most/all predictions, but the function must not crash.
    Results are still appended to batch_data with empty prediction lists, so tasks
    are created (with no pre-annotations) and total_tasks_created > 0.
    """
    print("Testing conf_threshold=0.999 (extreme filtering)...")
    proj_id = yolo_to_labelstudio(
        results_seg,
        task_type="segmentation",
        projectID=None,
        port=LS_PORT,
        project_title="Test-Adapter-HighThreshold",
        conf_threshold=0.999,
    )
    assert isinstance(proj_id, int) and proj_id > 0
    print(f"  test_conf_threshold_filtering: PASS (project ID {proj_id})")
    return proj_id


def test_invalid_task_type(results_seg):
    """task_type='invalid' must raise ValueError immediately, before any network call."""
    print("Testing ValueError for invalid task_type...")
    raised = False
    try:
        yolo_to_labelstudio(results_seg, task_type="invalid")
    except ValueError as exc:
        raised = True
        assert "invalid" in str(exc).lower() or "task_type" in str(exc).lower(), \
            f"ValueError message unexpected: {exc}"
    assert raised, "Expected ValueError was not raised for task_type='invalid'"
    print("  test_invalid_task_type: PASS")


def test_skips_results_without_masks(client, results_bbox_no_masks):
    """
    When task_type='segmentation' but all results have masks=None, every result is
    skipped by the per-result guard in adapters.py (line 82-83). batch_data stays
    empty, and import_preannotated_tasks_batch raises ValueError (total_tasks_created==0).

    This test documents and asserts that real behaviour. If future code should handle
    this case gracefully (e.g. skip the import call when batch_data is empty), the
    fix belongs in adapters.py, not here.
    """
    print("Testing ValueError when all results lack masks (seg task on masked-out results)...")
    raised = False
    try:
        yolo_to_labelstudio(
            results_bbox_no_masks,
            task_type="segmentation",
            projectID=None,
            port=LS_PORT,
            project_title="Test-Adapter-NoMasks",
        )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError when batch_data is empty (all results skipped)"
    print("  test_skips_results_without_masks: PASS")


def run_all_adapter_tests():
    print("=== Starting Adapter Tests ===")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    image_paths = glob.glob(str(IMAGE_DIR / "*.jpg")) + glob.glob(str(IMAGE_DIR / "*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")
    print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

    results_seg  = model.predict(source=image_paths, conf=0.3, verbose=False)
    results_bbox = results_seg   # seg results carry .boxes — valid for bbox task type

    class _MasksNoneWrapper:
        """Proxy that returns None for .masks, forwards all other attrs to the real result."""
        def __init__(self, real_result):
            self._real = real_result
        def __getattr__(self, name):
            if name == "masks":
                return None
            return getattr(self._real, name)

    results_bbox_no_masks = [_MasksNoneWrapper(r) for r in results_seg]

    client = LabelStudioClient(LS_PORT, API_KEY)
    created_project_ids = []

    try:
        created_project_ids.append(test_seg_new_project(client, results_seg))
        created_project_ids.append(test_bbox_new_project(client, results_bbox))
        created_project_ids.append(test_seg_existing_project(client, results_seg))
        created_project_ids.append(test_bbox_existing_project(client, results_bbox))
        created_project_ids.append(test_conf_threshold_filtering(client, results_seg))

        test_invalid_task_type(results_seg)   # no project created

        # Capture orphaned project (shell created before ValueError) and clean up
        ids_before = {e["ID"] for e in client.get_projects_summary()}
        test_skips_results_without_masks(client, results_bbox_no_masks)
        ids_after  = {e["ID"] for e in client.get_projects_summary()}
        created_project_ids.extend(ids_after - ids_before)

        print("\n✅ All adapter tests passed successfully.")

    finally:
        print("\n--- Cleaning up test projects ---")
        for pid in created_project_ids:
            try:
                resp = client.session.delete(f"{client.base_url}/api/projects/{pid}/")
                status = "OK" if resp.status_code == 204 else f"status {resp.status_code}"
                print(f"  Deleted project {pid}: {status}")
            except Exception as exc:
                print(f"  Error deleting project {pid}: {exc}")


if __name__ == "__main__":
    run_all_adapter_tests()
