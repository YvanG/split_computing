# Split Computing Project


## Install environment
```bash 
conda create -n split_computing
pip install ultralytics=8.0.227
```

## Model architecture
![splits](/assets/yolo8-splits.png)

## Run prediction
### Model
```bash 
python yolov8/yolo8_predict.py \
  --model_path checkpoints/100a-baseline/weights/last.pt \
  --split a
```
- model options:
  - checkpoints/100a-baseline/weights/last.pt  (our pre-trained model)
  - configs/yolo8/yolov8m.yaml  (new model with random weights)

- split options:
  - a
  - b
  - c

### Model - bottleneck
```bash 
python yolov8/yolo8_predict.py \
  --model_path checkpoints/101a-bottleneck_a16/weights/last.pt \
  --split a \
  --bottleneck
```


- split options:
  - a
  - b
  - c

- model options
  - split - a
    - checkpoints/101a-bottleneck_a16/weights/last.pt
    - checkpoints/102a-bottleneck_a32/weights/last.pt
    - checkpoints/103a-bottleneck_a64/weights/last.pt
    - checkpoints/111a-bottleneck_a16/weights/last.pt  (trained with frozen weights)
    - checkpoints/112a-bottleneck_a32/weights/last.pt  (trained with frozen weights)
    - checkpoints/113a-bottleneck_a64/weights/last.pt  (trained with frozen weights)
  
  - split - b
    - checkpoints/104a-bottleneck_b16/weights/last.pt
    - checkpoints/105a-bottleneck_b32/weights/last.pt
    - checkpoints/106a-bottleneck_b64/weights/last.pt
    - checkpoints/114a-bottleneck_b16/weights/last.pt  (trained with frozen weights)
    - checkpoints/115a-bottleneck_b32/weights/last.pt  (trained with frozen weights)
    - checkpoints/116a-bottleneck_b64/weights/last.pt  (trained with frozen weights)
  
  - split - c
    - checkpoints/107a-bottleneck_c16/weights/last.pt
    - checkpoints/108a-bottleneck_c32/weights/last.pt
    - checkpoints/109a-bottleneck_c64/weights/last.pt
    - checkpoints/117a-bottleneck_c16/weights/last.pt  (trained with frozen weights)
    - checkpoints/118a-bottleneck_c32/weights/last.pt  (trained with frozen weights)
    - checkpoints/119a-bottleneck_c64/weights/last.pt  (trained with frozen weights)


