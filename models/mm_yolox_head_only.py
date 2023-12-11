from mmdet.apis import DetInferencer

from models.utils import Identity

import mmcv

def mm_yolox_head_only(device='cuda:0'):
    config = 'C:\Work\split_computing\configs\yolox\yolox_s_8xb8-300e_coco.py'  # zmenit!!!!
    checkpoint = 'C:\Work\split_computing\checkpoints\yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'   # zmenit!!!
    model = DetInferencer(config, checkpoint, device='cuda:0')
    head = model
    tail = Identity()
    return head, tail


if __name__ == '__main__':
    import numpy as np

    head, tail = mm_yolox_head_only()
    rand_input = np.round(np.random.rand(640, 640, 3)*255).astype(np.uint8)
    output = head(rand_input)
    output2 = tail(output)
