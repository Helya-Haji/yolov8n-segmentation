from ultralytics import YOLO
import os

dataset_path = r"dataset"
data_yaml_path = r"dataset\data.yaml"

def train_segmentation():
    model = YOLO('yolov8n-seg.pt')  # segmentation model
    print("✅ Model loaded successfully")
    
    train_args = {
        'data': data_yaml_path,
        'epochs': 200,
        'imgsz': 640,
        'batch': 32,
        'device': 0,  # use GPU if available
        'project': r'runs\segment',
        'name': 'segmentation',
        'exist_ok': True,
        'save': True,
        'save_period': 10,
        'cache': False,
        'single_cls': True,
        'optimizer': 'SGD',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'plots': True,
        'lr0': 0.001,
        'lrf': 0.001,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'hsv_h':0.015,    
        'hsv_s':0.7,     
        'hsv_v':0.4,     
        'degrees':0.0,      
        'translate':0.1,  
        'scale':0.5,       
        'shear':0.0,       
        'perspective':0.0, 
        'flipud':0.0,    
        'fliplr':0.5,  
        'mosaic':1.0, 
        'mixup':0.0,     
        'patience':0,     
        'save':True,
        'cache':False,      
        'rect':True,      
        'cos_lr':True,     
        'close_mosaic':10, 
        'amp':True,   
        'overlap_mask':True,
        'mask_ratio':4,
        'val':True,
        'plots':True,
    }
    
    results = model.train(**train_args)
    return os.path.join('runs', 'segment', 'weights', 'best.pt')

if __name__ == "__main__":
    trained_model_path = train_segmentation()
    
