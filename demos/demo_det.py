import os
import sys
from prelabel.label_studio_utils import push_yolo_to_labelstudio, LabelStudioClient, generate_yolo_labels

IMGPATH = 'assets/images_YOLO/person_and_bike_188.png'
SAM3_CLASS_NAMES = ["Traffic-sign", "Person", "Car", "Person carrying a handbag"]

mode = sys.argv[1] if len(sys.argv) > 1 else sys.exit("Usage: python script.py [yolo|sam3]")

if mode == "yolo":
    from ultralytics import YOLO
    model = YOLO("yolov8n-seg.pt")
    results = model(IMGPATH)
    class_names = model.names
    r = results[0]

elif mode == "sam3":
    if not os.path.exists("sam3.pt"):
        sys.exit("Error: 'sam3.pt' not found. Please ensure the model file exists.")
    from ultralytics.models.sam import SAM3SemanticPredictor
    predictor = SAM3SemanticPredictor(overrides=dict(conf=0.5, task="segment", mode="predict", model="sam3.pt", half=True, save=False, imgsz=640))
    predictor.set_image(IMGPATH)
    class_names = SAM3_CLASS_NAMES
    r = predictor(text=class_names)[0]

else:
    sys.exit(f"Unknown mode '{mode}'. Choose 'yolo' or 'sam3'.")

yolo_labels = generate_yolo_labels(r, class_names)
ls = LabelStudioClient('8080', os.getenv('LABELSTUDIO_TOKEN'))
proj_id = ls.create_cv_project_BrushLabels(title=f"Demo - {mode.upper()}", labels=yolo_labels)
push_yolo_to_labelstudio(yolo_result=r, 
                         img_path=IMGPATH, 
                         port=8080, 
                         api_key=os.getenv('LABELSTUDIO_TOKEN'), 
                         project_id=proj_id, 
                         task_type="segmentation", 
                         conf_threshold=0,
                         )
