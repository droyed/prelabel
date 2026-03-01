import os
import sys
from ultralytics import YOLO
import glob
from prelabel.adapters import setup_project_with_yolo_results

model = YOLO("yolov8n-seg.pt")

imgsdir = "assets/images_YOLO"

# Load all images in the directory using glob with extension .jpg or .png
imgs = glob.glob(os.path.join(imgsdir, "*.jpg")) + glob.glob(os.path.join(imgsdir, "*.png"))
results = model.predict(source=imgs, conf=0.3, verbose=False)
proj_id = setup_project_with_yolo_results(results, task_type="segmentation", project_title="Seg Project")
proj_id = setup_project_with_yolo_results(results, task_type="bbox", project_title="Bbox Project")

#--------------------------------------------------------------
IMGPATH = 'assets/images_YOLO/person_337.png'

SAM3_CLASS_NAMES = ["person with white shirt"]

if not os.path.exists("sam3.pt"):
    sys.exit("Error: 'sam3.pt' not found. Please ensure the model file exists.")

if not os.path.exists("bpe_simple_vocab_16e6.txt.gz"):
    sys.exit("Error: 'bpe_simple_vocab_16e6.txt.gz' not found. Please ensure the file exists.")

from ultralytics.models.sam import SAM3SemanticPredictor
predictor = SAM3SemanticPredictor(overrides=dict(conf=0.5, task="segment", mode="predict", model="sam3.pt", half=True, save=False, imgsz=640), bpe_path="bpe_simple_vocab_16e6.txt.gz")
predictor.set_image(IMGPATH)
class_names = SAM3_CLASS_NAMES
results = predictor(text=class_names)
proj_id = setup_project_with_yolo_results(results, task_type="segmentation", project_title="Seg Project - SAM3")
