import requests
import os
import base64
import mimetypes
import uuid
import cv2
import numpy as np
import sys
from label_studio_converter.brush import mask2rle



class LabelStudioClient:
    def __init__(self, port, api_key):
        """Initializes the client and sets up a persistent request session."""
        self.base_url = f"http://localhost:{port}"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        })

    def project_exists(self, project_id, raise_on_missing=True):
        """
        Verifies if a specific project ID exists on the instance.
        
        Args:
            project_id (int): The ID of the project to check.
            raise_on_missing (bool): If True, raises a ValueError if not found. 
                                     If False, simply returns False.
                                     
        Returns:
            bool: True if the project exists, False otherwise.
        """
        url = f"{self.base_url}/api/projects/{project_id}/"
        response = self.session.get(url)
        
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            if raise_on_missing:
                raise ValueError(
                    f"❌ Error: Project ID {project_id} does not exist. "
                    f"Please verify the ID or create a new project on {self.base_url} first."
                )
            return False
            
        # Catch any other unexpected API errors (like 401 Unauthorized)
        response.raise_for_status()
        return False
            
    def delete_all_projects(self):
        """Deletes all projects currently hosted on the Label Studio instance."""
        response = self.session.get(f"{self.base_url}/api/projects/")
        response.raise_for_status()
        
        projects = response.json().get('results', [])
        deleted_count = 0
        
        for project in projects:
            project_id = project['id']
            del_response = self.session.delete(f"{self.base_url}/api/projects/{project_id}/")
            if del_response.status_code == 204:
                print(f"Deleted project ID: {project_id}")
                deleted_count += 1
                
        return deleted_count

    def create_cv_project(self, title, labels):
        """Creates a new project configured for bounding box image annotation."""
        label_tags = "".join([f'<Label value="{label}"/>' for label in labels])
        label_config = f"""
        <View>
          <Image name="image" value="$image"/>
          <RectangleLabels name="label" toName="image">
            {label_tags}
          </RectangleLabels>
        </View>
        """
        
        payload = {
            "title": title,
            "label_config": label_config,
        }
        
        response = self.session.post(f"{self.base_url}/api/projects/", json=payload)
        response.raise_for_status()
        return response.json()['id']

    def import_local_images(self, project_id, image_directory):
        """Uploads local images to a project. Requires LOCAL_FILES_SERVING_ENABLED=true."""
        # Create a separate header dictionary because multipart/form-data 
        # requires requests to set the Content-Type automatically with a boundary.
        upload_headers = {"Authorization": self.session.headers["Authorization"]}
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        imported_count = 0
        
        for filename in os.listdir(image_directory):
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                file_path = os.path.join(image_directory, filename)
                
                with open(file_path, 'rb') as f:
                    files = {'file': (filename, f)}
                    response = requests.post(
                        f"{self.base_url}/api/projects/{project_id}/import",
                        headers=upload_headers,
                        files=files
                    )
                    
                if response.status_code in [200, 201]:
                    imported_count += 1
                    
        return imported_count

    def _create_project_from_tag(self, title, labels, tag_name):
        """Internal helper to create CV projects dynamically based on the tag name."""
        label_tags = "".join([f'<Label value="{label}"/>' for label in labels])
        label_config = f"""
        <View>
          <Image name="image" value="$image"/>
          <{tag_name} name="label" toName="image">
            {label_tags}
          </{tag_name}>
        </View>
        """
        
        payload = {
            "title": title,
            "label_config": label_config.strip(),
        }
        
        response = self.session.post(f"{self.base_url}/api/projects/", json=payload)
        response.raise_for_status()
        
        project_id = response.json()['id']
        print(f"Created '{title}' ({tag_name}) with ID: {project_id}")
        return project_id

    def create_bbox_project(self, title, labels):
        """Creates an object detection project using bounding boxes."""
        return self._create_project_from_tag(title, labels, "RectangleLabels")

    def create_polygon_project(self, title, labels):
        """Creates a precise object segmentation project using polygons."""
        return self._create_project_from_tag(title, labels, "PolygonLabels")

    def create_brush_project(self, title, labels):
        """Creates a semantic segmentation project using brush masks."""
        return self._create_project_from_tag(title, labels, "BrushLabels")    
    

    def export_annotations(self, project_id, export_type, output_path):
        """
        Core method to export annotations. 
        Downloads the export package (ZIP) to the specified path.
        """
        url = f"{self.base_url}/api/projects/{project_id}/export?exportType={export_type}"
        response = self.session.get(url)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Exported project {project_id} as {export_type} to {output_path}")
        return output_path

    def export_bbox_yolo(self, project_id, output_path="yolo_bboxes.zip"):
        """Exports bounding box annotations in YOLO format."""
        return self.export_annotations(project_id, "YOLO", output_path)

    def export_polygon_coco(self, project_id, output_path="coco_polygons.zip"):
        """Exports polygon annotations in COCO format."""
        return self.export_annotations(project_id, "COCO", output_path)

    def export_brush_png(self, project_id, output_path="png_masks.zip"):
        """Exports brush mask annotations as rasterized PNG images."""
        return self.export_annotations(project_id, "PNG", output_path)

    def list_projects_summary(self):
        """Fetches all projects and prints a tabular summary of their status."""
        response = self.session.get(f"{self.base_url}/api/projects/")
        response.raise_for_status()
        
        projects = response.json().get('results', [])
        
        if not projects:
            print("No projects found on the Label Studio instance.")
            return
            
        # Define table headers and fixed-width column formatting
        headers = ["ID", "Title", "Classes", "Tasks", "Annotated", "Progress", "Annots", "Created Date"]
        row_format = "{:<4} | {:<22} | {:<7} | {:<6} | {:<9} | {:<8} | {:<6} | {:<12}"
        
        print("\n" + "=" * 90)
        print(row_format.format(*headers))
        print("-" * 90)
        
        for p in projects:
            pid = p.get('id', 'N/A')
            title = p.get('title', 'Untitled')[:22] # Truncate long titles to fit
            
            # Dynamically extract the number of classes from the parsed XML config
            num_classes = 0
            config = p.get('parsed_label_config', {})
            for key, value in config.items():
                if isinstance(value, dict) and 'labels' in value:
                    num_classes += len(value['labels'])
                    
            # Task and annotation metrics
            total_tasks = p.get('task_number', 0)
            annotated_tasks = p.get('num_tasks_with_annotations', 0)
            total_annots = p.get('total_annotations_number', 0)
            
            # Calculate progress percentage safely
            if total_tasks > 0:
                progress = f"{(annotated_tasks / total_tasks) * 100:.1f}%"
            else:
                progress = "0.0%"
                
            # Slice the ISO datetime string to just get the YYYY-MM-DD
            created_at = p.get('created_at', 'Unknown')[:10]
            
            print(row_format.format(
                pid, title, num_classes, total_tasks, annotated_tasks, progress, total_annots, created_at
            ))
            
        print("=" * 90 + "\n")
    
    def cleanup_empty_projects(self):
        """
        Iterates through all projects and deletes any that have exactly zero tasks.
        Returns a list of the deleted project titles.
        """
        response = self.session.get(f"{self.base_url}/api/projects/")
        response.raise_for_status()
        
        projects = response.json().get('results', [])
        deleted_count = 0
        deleted_titles = []
        
        for p in projects:
            project_id = p.get('id')
            total_tasks = p.get('task_number', 0)
            
            # Target only projects that have zero tasks uploaded to them
            if total_tasks == 0:
                title = p.get('title', 'Untitled')
                del_response = self.session.delete(f"{self.base_url}/api/projects/{project_id}/")
                
                if del_response.status_code == 204:
                    print(f"Deleted empty project: '{title}' (ID: {project_id})")
                    deleted_count += 1
                    deleted_titles.append(title)
                else:
                    print(f"Failed to delete project '{title}' (ID: {project_id}). Status: {del_response.status_code}")
                    
        print(f"\nCleanup complete. Removed {deleted_count} empty projects.")
        return deleted_titles


    def import_preannotated_task(self, project_id, image_path, ls_predictions, model_version="yolo-model"):
        """
        Encodes an image to Base64 and pushes it along with its predictions 
        to Label Studio in a single API call.
        """
        # 0. Fail-fast: Validates project existence (raises exception if missing)
        self.project_exists(project_id, raise_on_missing=True)

        # 1. Convert the image to a Base64 Data URI
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/png" 
        image_data_uri = f"data:{mime_type};base64,{img_b64}"

        # 2. Build the unified Label Studio Import Payload
        payload = [
            {
                "data": {
                    "image": image_data_uri  # Maps to <Image value="$image"/>
                },
                "predictions": [
                    {
                        "model_version": model_version,
                        "result": ls_predictions
                    }
                ]
            }
        ]

        # 3. Push it all at once to the Import endpoint
        import_url = f"{self.base_url}/api/projects/{project_id}/import"
        response = self.session.post(import_url, json=payload)
        response.raise_for_status()
        
        import_data = response.json()
        task_count = import_data.get('task_count', 0)
        
        if task_count == 0:
            raise ValueError(f"Import processed, but no tasks were created. Response: {import_data}")
            
        print(f"✅ Success! Task and predictions imported for {os.path.basename(image_path)}.")
        
        project_url = f"{self.base_url}/projects/{project_id}/data"
        print(f"🔗 View and verify your imported data here: {project_url}")
        
        return import_data


def extract_ls_predictions(yolo_result, task_type="segmentation", from_name="tag", to_name="image", conf_threshold=0.0):
    """
    Converts a single YOLO Result object into a list of Label Studio prediction dictionaries.
    
    Args:
        yolo_result: A single ultralytics.engine.results.Result object.
        task_type (str): "segmentation" (outputs BrushLabels) or "detection" (outputs RectangleLabels).
        from_name (str): The name of the labeling tag in your LS XML config.
        to_name (str): The name of the object tag in your LS XML config.
        conf_threshold (float): Minimum confidence score to include the prediction.
        
    Returns:
        list: A list of dictionaries formatted for Label Studio regions.
    """
    orig_height, orig_width = yolo_result.orig_shape
    names = yolo_result.names
    prediction_results = []
    
    # Check if the model detected anything
    if not hasattr(yolo_result, 'boxes') or yolo_result.boxes is None:
        return prediction_results
        
    boxes = yolo_result.boxes
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.int().cpu().numpy()
    
    # Pre-fetch masks if segmentation is requested
    if task_type == "segmentation":
        if not hasattr(yolo_result, 'masks') or yolo_result.masks is None:
            print("Warning: task_type is 'segmentation' but no masks found in YOLO result.")
            masks = []
        else:
            masks = yolo_result.masks.data.cpu().numpy()
            
    for i in range(len(boxes)):
        conf = float(confs[i])
        
        # Filter out low-confidence predictions
        if conf < conf_threshold:
            continue
            
        class_name = names[clss[i]].capitalize()
        region_id = str(uuid.uuid4())[:8]
        
        # Base dictionary shared by all Label Studio prediction types
        base_region = {
            "id": region_id,
            "origin": "manual",
            "to_name": to_name,
            "from_name": from_name,
            "image_rotation": 0,
            "original_width": orig_width,
            "original_height": orig_height,
            "score": conf,
            "meta": {
                "text": [f"Conf: {conf:.2%}"]
            }
        }
        
        # Handle Masks (BrushLabels via RLE)
        if task_type == "segmentation":
            mask = masks[i]
            
            # Convert to uint8 if not already            
            if mask.dtype == np.bool_:
                mask = mask.astype(np.uint8)
                
            assert mask.dtype == np.uint8
            
            # Resize and threshold
            mask_resized = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
            mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255
            
            seg_region = base_region.copy()
            seg_region.update({
                "type": "brushlabels",
                "value": {
                    "format": "rle",
                    "rle": mask2rle(mask_uint8),
                    "brushlabels": [class_name]
                }
            })
            prediction_results.append(seg_region)
            
        # Handle Bounding Boxes (RectangleLabels)
        elif task_type == "detection":
            # YOLO normalized xyxy: [x_min, y_min, x_max, y_max] from 0.0 to 1.0
            box_n = boxes.xyxyn[i].cpu().numpy() 
            x_min, y_min, x_max, y_max = box_n
            
            det_region = base_region.copy()
            det_region.update({
                "type": "rectanglelabels",
                "value": {
                    # Label Studio expects bounding boxes as percentages (0-100)
                    "x": float(x_min * 100),
                    "y": float(y_min * 100),
                    "width": float((x_max - x_min) * 100),
                    "height": float((y_max - y_min) * 100),
                    "rotation": 0,
                    "rectanglelabels": [class_name]
                }
            })
            prediction_results.append(det_region)
            
    return prediction_results


def check_label_studio_running(port, timeout=5):
    """Check if Label Studio is running and accessible.

    Makes a GET request to the Label Studio health endpoint to verify
    that the service is running on the specified port.

    Args:
        port: The port number where Label Studio is running.
        timeout: Request timeout in seconds. Defaults to 5.

    Returns:
        bool: True if Label Studio is running and responds with status 200,
              False otherwise.

    Prints:
        A status message indicating whether Label Studio is running,
        or instructions to start it if not running.
    """
    host="http://localhost"
    url = f"{host}:{port}/health"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✓ Label Studio is running on port {port}")
            return True
    except requests.exceptions.ConnectionError:
        print(f"✗ Label Studio is NOT running on port {port}. Please run: label-studio start --port {port}")
    except requests.exceptions.Timeout:
        print(f"✗ Label Studio health check timed out on port {port}")
    return False


def push_yolo_to_labelstudio(yolo_result, img_path, port, api_key, project_id, task_type="segmentation", conf_threshold=0.5):
    """
    Extracts predictions from a YOLO result object and pushes them along 
    with the source image to a Label Studio project.
    
    Args:
        yolo_result: A single result object from a YOLO model (e.g., results[0]).
        img_path (str): Local path to the image file.
        port (int): The Label Studio instance port.
        api_key (str): Your Label Studio API token.
        project_id (int): The destination project ID.
        task_type (str): "segmentation" or "detection".
        conf_threshold (float): Minimum confidence threshold for predictions.
        
    Returns:
        dict: The summary dictionary returned by the Label Studio import endpoint.
    """
    # Guard at the top of your Step 2 script
    if not check_label_studio_running(port=port):
        sys.exit(1)
    
    print(f"Processing YOLO results for {os.path.basename(img_path)}...")
    
    # 1. Extract the prediction regions
    ls_predictions = extract_ls_predictions(
        yolo_result=yolo_result,
        task_type=task_type,
        conf_threshold=conf_threshold
    )

    print(f"Extracted {len(ls_predictions)} regions. Pushing to API...")

    # 2. Initialize client and push to Label Studio
    ls = LabelStudioClient(port, api_key)
    import_summary = ls.import_preannotated_task(project_id, img_path, ls_predictions)
    
    return import_summary
