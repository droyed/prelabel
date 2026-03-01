import os
from prelabel.label_studio_utils import (
    LabelStudioClient,
    generate_yolo_labels_from_classnames,
    extract_ls_predictions,
)

def setup_project_with_yolo_results(
    results,
    task_type="segmentation",
    projectID=None,
    port=8080,
    labelstudio_token_name="LABELSTUDIO_TOKEN",
    project_title="Demo",
    model_version="yolov8n-model-v1",
    batch_size=25,
    conf_threshold=0.0
):
    """
    Set up a Label Studio project with YOLO prediction results for either Segmentation or Bounding Box tasks.

    If projectID is None, creates a new project (BrushLabels for segmentation, RectangleLabels for bbox) 
    and imports tasks. If projectID is provided, uses the existing project and imports tasks into it.

    Args:
        results: List of YOLO Result objects (e.g. from model.predict(...)).
        task_type (str): The type of annotation task. Options: ["segmentation", "bbox"]. 
            Defaults to "segmentation".
        projectID (int, optional): Existing project ID. If None, a new project is created via 
            create_cv_project_generic. If provided, pre-annotations are imported into this project.
        port (int): Label Studio server port. Default 8080.
        labelstudio_token_name (str): Environment variable name for the Label Studio 
            API token. Default "LABELSTUDIO_TOKEN".
        project_title (str): Title for the new project (only used when projectID is None). 
            Default "Demo".
        model_version (str): Model version string for predictions. Default "yolov8n-model-v1".
        batch_size (int): Number of tasks per import batch. Default 25.
        conf_threshold (float): Minimum confidence for including predictions. Default 0.0.

    Returns:
        int: The project ID (newly created or the provided projectID).

    Raises:
        ValueError: If an invalid `task_type` is provided.

    Examples:
        # 1. Segmentation Task (New Project)
        >>> model = YOLO("yolov8n-seg.pt")
        >>> results = model.predict(source=imgs, conf=0.3)
        >>> proj_id = setup_project_with_yolo_results(
        ...     results,
        ...     task_type="segmentation",
        ...     project_title="Seg Project"
        ... )

        # 2. Bounding Box Task (Existing Project ID 10)
        >>> model = YOLO("yolov8n.pt") # or seg model
        >>> results = model.predict(source=imgs, conf=0.5)
        >>> proj_id = setup_project_with_yolo_results(
        ...     results,
        ...     task_type="bbox",
        ...     projectID=10
        ... )
    """
    if task_type not in ["segmentation", "bbox"]:
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'segmentation' or 'bbox'.")

    batch_data = []    
    class_names_results = set()
    
    # Map task types to Label Studio label types and YOLO attributes
    config = {
        "segmentation": {"ls_type": "BrushLabels", "yolo_attr": "masks"},
        "bbox": {"ls_type": "RectangleLabels", "yolo_attr": "boxes"}
    }
    
    current_config = config[task_type]

    for result in results:
        # Safety check: Ensure the result actually contains the data we need (masks or boxes)
        # getattr is used to dynamically check result.masks or result.boxes based on config
        if getattr(result, current_config["yolo_attr"]) is None:
            continue

        # 1. Extract the prediction regions
        predictions, class_names_result = extract_ls_predictions(
            yolo_result=result,
            task_type=task_type,
            conf_threshold=conf_threshold
        )
        class_names_results.update(class_names_result)

        batch_data.append({
            "image_path": result.path,
            "predictions": predictions
        })

    ls = LabelStudioClient(port, os.getenv(labelstudio_token_name))
    class_names = list(class_names_results)
    yolo_labels = generate_yolo_labels_from_classnames(class_names)

    if projectID is None:
        # Create project with dynamic label type
        proj_id = ls.create_cv_project_generic(
            title=project_title, 
            labels=yolo_labels, 
            label_type=current_config["ls_type"]
        )        
    else:
        proj_id = projectID
        ls.project_exists(proj_id, raise_on_missing=True)

    ls.import_preannotated_tasks_batch(
        project_id=proj_id,
        batch_data=batch_data,
        model_version=model_version,
        batch_size=batch_size
    )
    return proj_id

