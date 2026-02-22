import os
from ultralytics import YOLO

imgpath = 'assets/images_YOLO/person_207.png'

model = YOLO("yolov8n-seg.pt")  

# Predict with the model
results = model(imgpath)  # predict on an image

from prelabel.label_studio_utils import push_yolo_to_labelstudio

push_yolo_to_labelstudio(yolo_result=results[0],
                            img_path=imgpath, 
                            port=8080, 
                            api_key=os.getenv('LABELSTUDIO_TOKEN'), 
                            project_id=127, 
                            task_type="segmentation", 
                            conf_threshold=0.5,
                            )
