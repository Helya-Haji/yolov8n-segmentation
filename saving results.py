from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO(r"runs\segment\femur_segmentation\weights\best.pt")

def segment_and_save(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"cant open the image: {image_path}")
        return
    
    h, w = img.shape[:2]

    results = model.predict(source=img, imgsz=640, conf=0.4, save=False, verbose=False)

    output = np.ones((h, w, 3), dtype=np.uint8) * 255

    for r in results:
        if r.masks is not None:
            for mask in r.masks.xy:
                pts = np.array(mask, dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                mask_img = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_img, [pts], 255)

                obj = cv2.bitwise_and(img, img, mask=mask_img)
                inv_mask = cv2.bitwise_not(mask_img)
                bg = cv2.bitwise_and(output, output, mask=inv_mask)
                output = cv2.add(bg, obj)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, output)
    print(f"output saved: {save_path}")

if __name__ == "__main__":
    input_folder = r"images"
    output_folder = r"images\results"

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            segment_and_save(input_path, output_path)
