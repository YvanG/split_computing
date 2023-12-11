import os
from PIL import Image

import mmcv

from mmdet.apis import init_detector, inference_detector
from mmdet.apis import DetInferencer

config = 'configs/yolox/yolox_s_8xb8-300e_coco.py'
# config = 'configs/faster_rcnn/faster-rcnn_r50_fpn_ms-3x_coco.py'
checkpoint = 'models/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
# checkpoint = 'models/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

inferencer = DetInferencer(config, checkpoint, device='cuda:0')
model = init_detector(config, checkpoint, device='cuda:0')

img = r'C:\Work\Amitia\Object_tracking\data\video_20220310-1253_0.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
results = inferencer(img)

print()
