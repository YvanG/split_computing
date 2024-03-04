# Split Computing


## Install environment
```bash 
conda create -n split_computing
pip install ultralytics=8.0.227
```

## YOLOv8 Split Locations Diagram
![splits](/assets/yolo8-diagram.png)


## Data preparation
All models are trained and validated on [COCO](https://cocodataset.org/#home) dataset.
Dataset should be in YOLOv8 format:
```
  coco 
  │
  └───images
  │   │
  │   └───val2017
  │   │    │   image_01.jpg
  │   │    │   ...
  │   │
  │   └───train2017
  │        │   image_02.jpg
  │        │   ...
  │   
  └───labels
  │   │
  │   └───val2017
  │   │    │   image_01.txt
  │   │    │   ...
  │   │
  │   └───train2017
  │        │   image_02.txt
  │        │   ...
  │   
  └───annotations     # necessary only for evaluation
  │   │ 
  │   │   instances_val2017.json    
```

Some of the bottlenecks reduce input image 4 times. 
During validation, it is necessary to ensure that the size of all images is divisible by 64.  
We have ensured this by resizing and padding all validation images.
```bash 
python yolov8/data_preparation.py \
  --dataset_root path/to/coco
```

## Train model
```bash 
python yolov8/yolo8_train.py \
  --model_name ../configs/yolo8/models/yolov8m_early_bn-1.yaml \
  --data_path ../configs/yolo8/models/coco.yaml \
  --workers 4 \
  --epochs 36 \
  --optimizer "SGD" \
  --lr0 0.005 \
  --batch 16 \
  --yolo_checkpoint ../yolo_weights/yolov8m.pt
```

## Evaluation
```bash 
python yolov8/yolo8_eval.py \
  --dataset_path ../configs/yolo8/models/coco.yaml \
  --checkpoint_path path/to/checkpoint.pt
```
