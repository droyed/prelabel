"""
prelabel — YOLO/SAM to Label Studio annotation bridge toolkit.

Core layer (requires only `requests`):
    from prelabel import LabelStudioClient

ML layer (requires [ml] extras: ultralytics, opencv-python, numpy, label-studio-converter):
    from prelabel.label_studio_utils import push_yolo_to_labelstudio
"""

__version__ = "0.1.0"

from prelabel.label_studio_utils import LabelStudioClient

__all__ = [
    "LabelStudioClient",
    "push_yolo_to_labelstudio",
    "__version__",
]
