from ultralytics import YOLO

model = YOLO(r"runs\segment\femur_segmentation\weights\best.pt")

results = model.predict(
    source=r"images", 
    save=True,
    project=r"images",
    name="inference",  
    exist_ok=True
)

print("saved")