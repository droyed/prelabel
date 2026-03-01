import os
import tempfile


# 1. Setup: Generate dummy images for the demo
temp_dir = tempfile.mkdtemp()
dummy_img_path = os.path.join(temp_dir, "demo_image.png")
with open(dummy_img_path, "wb") as f:
    f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')

from prelabel import LabelStudioClient

LS_PORT = int(os.getenv("LABELSTUDIO_PORT", 8080))
API_KEY = os.getenv("LABELSTUDIO_TOKEN", "YOUR_API_KEY")

def workflow_pre_annotation():
    '''
    Workflow 1: AI-Assisted Bounding Box Annotation

    This workflow simulates an active learning pipeline. It creates a bounding box project,
    runs inference on local images (simulated here with dummy predictions), pushes the images
    and predictions to Label Studio in a single pass, and finally exports the dataset in
    YOLO format for retraining.
    '''

    print("--- Starting AI-Assisted Annotation Workflow ---")
    ls = LabelStudioClient(LS_PORT, API_KEY)
    
    # 2. Create a targeted Computer Vision project
    labels = ["Vehicle", "Pedestrian", "Traffic Light"]
    project_id = ls.create_bbox_project("Smart City - Detection", labels)
    
    # 3. Simulate a model inference loop (e.g., YOLO predicting a 'Vehicle')
    # In a real script, you would loop through os.listdir(temp_dir) and call your YOLO model
    print(f"Running inference and uploading to Project {project_id}...")
    
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
    
    # Push image and predictions atomically
    ls.import_preannotated_task(project_id, dummy_img_path, ls_predictions)
    
    # 4. Print summary to verify the task is ready for human review
    ls.list_projects_summary()
    
    # 5. Export to YOLO format (Usually done AFTER human review is complete)
    export_path = ls.export_bbox_yolo(project_id, output_path="smart_city_yolo.zip")
    print(f"Workflow Complete. Dataset ready for training at: {export_path}")


def workflow_bulk_ingestion(image_directory):
    '''
    Workflow 2: Bulk Data Ingestion and Maintenance
    
    This workflow focuses on data management. It creates a highly precise polygon 
    segmentation project, bulk-imports an entire directory of local images, runs a 
    cleanup of any empty test projects lying around your instance, and exports the data to COCO format.
    
    Note: For import_local_images to work, your local Label Studio instance must be 
    started with the environment variable LOCAL_FILES_SERVING_ENABLED=true.
    '''
    
    print("\n--- Starting Bulk Ingestion & Maintenance Workflow ---")
    ls = LabelStudioClient(LS_PORT, API_KEY)
    
    # 1. Housekeeping: Clean up any empty projects from previous failed runs
    print("Running initial cleanup...")
    ls.cleanup_empty_projects()
    
    # 2. Create an Instance Segmentation project
    labels = ["Tumor", "Healthy Tissue"]
    project_id = ls.create_polygon_project("Medical Imaging - Segmentation", labels)
    
    # 3. Bulk import all valid images from a local directory
    print(f"Importing images from {image_directory}...")
    imported_count = ls.import_local_images(project_id, image_directory)
    print(f"Successfully staged {imported_count} images for annotation.")
    
    # 4. View overall instance status
    ls.list_projects_summary()
    
    # 5. Export annotations to COCO JSON format (ideal for Mask R-CNN)
    export_path = ls.export_polygon_coco(project_id, output_path="medical_seg_coco.zip")
    print(f"Workflow Complete. COCO dataset exported to: {export_path}")

    

if __name__ == "__main__":
    workflow_pre_annotation()
    workflow_bulk_ingestion(temp_dir)