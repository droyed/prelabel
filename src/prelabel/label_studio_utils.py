import requests
import os
import base64
import mimetypes
import uuid
import numpy as np
from label_studio_converter.brush import mask2rle
from typing import List, Tuple, Union, Dict
import imtools


def generate_label_config(labels: list[dict], label_type: str = "BrushLabels") -> str:
    """
    Generates a Label Studio configuration string from a list of label dictionaries.
    
    Args:
        labels: A list of dictionaries containing 'name' and 'color' keys.
        label_type: The Label Studio tag to wrap the labels in (e.g., BrushLabels, PolygonLabels).
        
    Returns:
        A formatted XML string representing the label configuration.
    """
    # 1. Generate the individual <Label .../> strings
    label_tags = "".join([
        f'<Label value="{label["name"]}" background="{label["color"]}"/>' 
        for label in labels
    ])
    
    # 2. Insert them into the main configuration template
    label_config = f"""
<View>
  <Image name="image" value="$image" zoom="true"/>
  <{label_type} name="tag" toName="image">
    
    
  {label_tags}</{label_type}>
</View>        
"""
    
    # .strip() removes any accidental leading/trailing newlines
    return label_config.strip() 


class LabelStudioClient:
    def __init__(self, port, api_key):
        """Initializes the client and sets up a persistent request session."""
        check_label_studio_running(port, raise_on_error=True)
            
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


    def create_cv_project_generic(self, title, labels, label_type):
        """
        Creates a new computer vision project in Label Studio using BrushLabels.
        
        Args:
            title (str): The title of the new project (e.g., 'New Project 1').
            labels (list[dict]): A list of dictionaries representing labels, 
                where each dictionary should have a 'name' and 'color' key.
                Example:
                [
                    {'name': 'Person', 'color': '#FFA39E'},
                    {'name': 'Car', 'color': '#D4380D'}
                ]
            label_type: str
                Must be "BrushLabels" or "RectangleLabels"
                
        Returns:
            int: The ID of the newly created project.
        """
        label_config = generate_label_config(labels, label_type=label_type)
        
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
    
    def get_projects_summary(self) -> list[dict]:
        """ Fetches all projects and returns a targeted summary of each matching the table format in a list of dictionaries. """
        response = self.session.get(f"{self.base_url}/api/projects/")
        response.raise_for_status()
        
        projects = response.json().get('results', [])
        summary = []
        
        if not projects:
            return summary
            
        for p in projects:
            pid = p.get('id')
            if pid is None:
                continue
                
            title = p.get('title', 'Untitled')[:22] # truncate like the print function does
            
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
                progress_str = f"{(annotated_tasks / total_tasks) * 100:.1f}%"
            else:
                progress_str = "0.0%"
                
            # Slice the ISO datetime string to just get the YYYY-MM-DD
            created_at = p.get('created_at', 'Unknown')[:10]
            
            summary.append({
                "ID": pid,
                "Title": title,
                "Classes": num_classes,
                "Tasks": total_tasks,
                "Annotated": annotated_tasks,
                "Progress": progress_str,
                "Annots": total_annots,
                "Created Date": created_at
            })
            
        return summary

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


    def import_preannotated_tasks_batch(self, project_id, batch_data, model_version="yolo-model", batch_size=25):
        """
        Encodes multiple images to Base64 and pushes them along with their predictions 
        to Label Studio in batches to avoid payload size limits.
        
        Args:
            project_id (int/str): The Label Studio project ID.
            batch_data (list of dict): Expected format: [{'image_path': str, 'predictions': list}]
            model_version (str): Name/version of the model.
            batch_size (int): How many images to send per API call.
            
        Returns:
            int: Total number of tasks successfully created.
        """
        # 0. Fail-fast: Validates project existence        
        self.project_exists(project_id, raise_on_missing=True)

        import_url = f"{self.base_url}/api/projects/{project_id}/import"
        total_tasks_created = 0
        
        # Process in chunks to avoid massive JSON payloads
        for i in range(0, len(batch_data), batch_size):
            chunk = batch_data[i:i + batch_size]
            payload = []
            
            # 1 & 2. Convert images and build payload for this chunk
            for task_info in chunk:
                image_path = task_info['image_path']
                ls_predictions = task_info['predictions']
                
                try:
                    with open(image_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode('utf-8')
                except FileNotFoundError:
                    print(f"⚠️ Warning: Image not found at {image_path}. Skipping this image.")
                    continue
                    
                mime_type, _ = mimetypes.guess_type(image_path)
                mime_type = mime_type or "image/png" 
                image_data_uri = f"data:{mime_type};base64,{img_b64}"

                payload.append({
                    "data": {
                        "image": image_data_uri  # Maps to <Image value="$image"/>
                    },
                    "predictions": [
                        {
                            "model_version": model_version,
                            "result": ls_predictions
                        }
                    ]
                })

            if not payload:
                continue

            # 3. Push the chunk to the Import endpoint
            response = self.session.post(import_url, json=payload)
            response.raise_for_status()
            
            import_data = response.json()
            tasks_in_chunk = import_data.get('task_count', 0)
            total_tasks_created += tasks_in_chunk
            
            print(f"🔄 Uploaded batch {i // batch_size + 1}... ({tasks_in_chunk} tasks created)")

        if total_tasks_created == 0:
            raise ValueError("Import processed, but no tasks were created. Check your data format.")
            
        print(f"✅ Success! A total of {total_tasks_created} tasks and predictions were imported.")
        
        project_url = f"{self.base_url}/projects/{project_id}/data"
        print(f"🔗 View and verify your imported data here: {project_url}")
        
        return total_tasks_created
    
    
from ultralytics.utils.ops import scale_masks


def check_label_studio_running(port, timeout=5, raise_on_error=False):
    """Check if Label Studio is running and accessible.

    Makes a GET request to the Label Studio health endpoint to verify
    that the service is running on the specified port.

    Args:
        port: The port number where Label Studio is running.
        timeout: Request timeout in seconds. Defaults to 5.
        raise_on_error: If True, raises a ConnectionError instead of returning False.

    Returns:
        bool: True if Label Studio is running and responds with status 200,
              False otherwise (if raise_on_error is False).

    Raises:
        ConnectionError: If raise_on_error is True and Label Studio is not accessible.

    Prints:
        A status message indicating whether Label Studio is running,
        or instructions to start it if not running.
    """
    host="http://localhost"
    url = f"{host}:{port}/health"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            #print(f"✓ Label Studio is running on port {port}")
            return True
        else:
            error_msg = f"✗ Label Studio responded with status code {response.status_code} on port {port}"
    except requests.exceptions.ConnectionError:
        error_msg = f"✗ Label Studio is NOT running on port {port}. Please run: label-studio start --port {port}"
    except requests.exceptions.Timeout:
        error_msg = f"✗ Label Studio health check timed out on port {port}"
        
    if raise_on_error:
        raise ConnectionError(error_msg)
    
    print(error_msg)
    return False


def rgb_to_hex(rgb: Union[List[int], Tuple[int, int, int]]) -> str:
    """
    Convert an RGB color list or tuple to a hex string.
    
    Args:
        rgb: A list or tuple of 3 integers representing Red, Green, and Blue components (0-255).
        
    Returns:
        Hexadecimal color string (e.g., '#ff0000').
    """
    if len(rgb) != 3:
        raise ValueError("Input must be a list or tuple of exactly 3 elements.")
        
    r, g, b = rgb
    
    # Clamp values between 0 and 255
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    return f"#{r:02x}{g:02x}{b:02x}"


def generate_yolo_labels_from_classnames(class_names: Union[Dict[int, str], List[str]]) -> List[Dict[str, str]]:
    """
    Generate a list of YOLO labels (name and color) from an Ultralytics YOLO Results object.

    Args:
        class_names: A dictionary or list mapping class indices to class names.

    Returns:
        A list of dictionaries containing 'name' and 'color' (hex string) for each user-selected class.
    """    
    colors = imtools.viz.colors.generate_colors(len(class_names), 'golden_ratio')

    yolo_labels = []
    for color, clsname in zip(colors, class_names):
        color_hex = rgb_to_hex(color)
        yolo_labels.append({'name': clsname, 'color': color_hex})
        
    return yolo_labels


def _validate_and_filter(yolo_result, conf_threshold):
    """
    Validates a YOLO result and returns filtered detections.
    Returns None if nothing valid is found.
    """
    if not hasattr(yolo_result, 'boxes') or yolo_result.boxes is None:
        return None

    boxes = yolo_result.boxes
    all_confs = boxes.conf.cpu().numpy()
    all_clss = boxes.cls.int().cpu().numpy()
    validmask = all_confs >= conf_threshold

    if not validmask.any():
        return None

    return validmask, all_confs[validmask], all_clss[validmask]


def _make_base_region(region_id, conf, orig_width, orig_height, from_name, to_name):
    """Builds the base Label Studio region dict shared by all task types."""
    return {
        "id": region_id,
        "origin": "manual",
        "to_name": to_name,
        "from_name": from_name,
        "image_rotation": 0,
        "original_width": orig_width,
        "original_height": orig_height,
        "score": conf,
        "meta": {"text": [f"Conf: {conf:.2%}"]},
    }

def extract_ls_bbox_predictions(yolo_result, from_name="tag", to_name="image", conf_threshold=0.0):
    """Converts YOLO results to Label Studio rectanglelabels predictions."""
    orig_height, orig_width = yolo_result.orig_shape
    names = yolo_result.names

    filtered = _validate_and_filter(yolo_result, conf_threshold)
    if filtered is None:
        return [], set()

    validmask, confs, clss = filtered
    boxes_n = yolo_result.boxes.xyxyn.cpu().numpy()[validmask]

    results = []
    class_names = set()
    for i, (conf, cls) in enumerate(zip(confs, clss)):
        conf = float(conf)
        class_name = names[cls]
        class_names.add(class_name)
        x_min, y_min, x_max, y_max = boxes_n[i]

        region = _make_base_region(str(uuid.uuid4())[:8], conf, orig_width, orig_height, from_name, to_name)
        region.update({
            "type": "rectanglelabels",
            "value": {
                "x": float(x_min * 100),
                "y": float(y_min * 100),
                "width": float((x_max - x_min) * 100),
                "height": float((y_max - y_min) * 100),
                "rotation": 0,
                "rectanglelabels": [class_name],
            },
        })
        results.append(region)
    return results, class_names


def extract_ls_segmentation_predictions(yolo_result, from_name="tag", to_name="image", conf_threshold=0.0):
    """Converts YOLO results to Label Studio brushlabels predictions."""
    orig_height, orig_width = yolo_result.orig_shape
    names = yolo_result.names

    filtered = _validate_and_filter(yolo_result, conf_threshold)
    if filtered is None:
        return [], set()

    validmask, confs, clss = filtered

    masks_4d = yolo_result.masks.data[validmask].unsqueeze(1)
    scaled_masks = scale_masks(masks_4d, (orig_height, orig_width)).squeeze(1)
    scaled_masks_uint8 = (scaled_masks.cpu().numpy() > 0.5) * np.uint8(255)

    results = []
    class_names = set()
    for i, (conf, cls) in enumerate(zip(confs, clss)):
        conf = float(conf)
        class_name = names[cls]
        class_names.add(class_name)

        region = _make_base_region(str(uuid.uuid4())[:8], conf, orig_width, orig_height, from_name, to_name)
        region.update({
            "type": "brushlabels",
            "value": {
                "format": "rle",
                "rle": mask2rle(scaled_masks_uint8[i]),
                "brushlabels": [class_name],
            },
        })
        results.append(region)
    return results, class_names


def extract_ls_predictions(yolo_result, task_type="bbox", **kwargs):
    """Unified entry point that dispatches to the appropriate extractor."""
    extractors = {
        "bbox": extract_ls_bbox_predictions,
        "segmentation": extract_ls_segmentation_predictions,
    }
    if task_type not in extractors:
        raise ValueError(f"Unknown task_type '{task_type}'. Choose from: {list(extractors.keys())}")
    return extractors[task_type](yolo_result, **kwargs)


