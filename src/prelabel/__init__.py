"""
prelabel — YOLO/SAM to Label Studio annotation bridge toolkit.

Core layer (requires only `requests`):
    from prelabel import LabelStudioClient

ML layer (requires [ml] extras: ultralytics, opencv-python, numpy, label-studio-converter):
    from prelabel.label_studio_utils import push_yolo_to_labelstudio
"""

__version__ = "1.0.0"

from .label_studio_utils import LabelStudioClient
from .adapters import setup_project_with_yolo_results

__all__ = [
    "LabelStudioClient",
    "setup_project_with_yolo_results",
]
