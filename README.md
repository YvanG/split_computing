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

- model options (trained models)
  - split - a
    - checkpoints/121a-bottleneck2_a16/weights/last.pt
    - checkpoints/122a-bottleneck2_a32/weights/last.pt
    - checkpoints/123a-bottleneck2_a64/weights/last.pt
    - 
    - checkpoints/131a-bottleneck2_a16/weights/last.pt  (trained with frozen weights)
    - checkpoints/132a-bottleneck2_a32/weights/last.pt  (trained with frozen weights)
    - checkpoints/133a-bottleneck2_a64/weights/last.pt  (trained with frozen weights)
    - 
  
  - split - b
    - checkpoints/124a-bottleneck2_b16/weights/last.pt
    - checkpoints/125a-bottleneck2_b32/weights/last.pt
    - checkpoints/126a-bottleneck2_b64/weights/last.pt
    - 
    - checkpoints/134a-bottleneck2_b16/weights/last.pt  (trained with frozen weights)
    - checkpoints/135a-bottleneck2_b32/weights/last.pt  (trained with frozen weights)
    - checkpoints/136a-bottleneck2_b64/weights/last.pt  (trained with frozen weights)
    - 
  
  - split - c
    - checkpoints/127a-bottleneck2_c16/weights/last.pt
    - checkpoints/128a-bottleneck2_c32/weights/last.pt
    - checkpoints/129a-bottleneck2_c64/weights/last.pt
    - 
    - checkpoints/137a-bottleneck2_c16/weights/last.pt  (trained with frozen weights)
    - checkpoints/138a-bottleneck2_c32/weights/last.pt  (trained with frozen weights)
    - checkpoints/139a-bottleneck2_c64/weights/last.pt  (trained with frozen weights)
    - 

- model options (random weights)
  - split - a
    - configs/yolo8/models/yolov8m_a16_bottleneck2.yaml
    - configs/yolo8/models/yolov8m_a32_bottleneck2.yaml
    - configs/yolo8/models/yolov8m_a64_bottleneck2.yaml
    - configs/yolo8/models/yolov8m_a_bottleneck3.yaml
 
  - split - b
    - configs/yolo8/models/yolov8m_b16_bottleneck2.yaml
    - configs/yolo8/models/yolov8m_b32_bottleneck2.yaml
    - configs/yolo8/models/yolov8m_b64_bottleneck2.yaml
    - configs/yolo8/models/yolov8m_b_bottleneck3.yaml

  - split - c
    - configs/yolo8/models/yolov8m_c16_bottleneck2.yaml
    - configs/yolo8/models/yolov8m_c32_bottleneck2.yaml
    - configs/yolo8/models/yolov8m_c64_bottleneck2.yaml
    - configs/yolo8/models/yolov8m_c_bottleneck3.yaml

