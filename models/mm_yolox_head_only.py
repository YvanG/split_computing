import numpy as np
from mmdet.apis import DetInferencer
from models.utils import Identity


class mm_yolox_model:
    def __init__(self, config, checkpoint, device='cuda:0'):
        self.model = DetInferencer(config, checkpoint, device=device)

    def __call__(self, x):
        x = self.model(x)
        boxes = np.array(x['predictions'][0]['bboxes'])
        confidences = np.expand_dims(np.array(x['predictions'][0]['scores']), axis=1)
        classes = np.expand_dims(np.array(x['predictions'][0]['labels']), axis=1)
        results = np.concatenate((boxes, confidences, classes), axis=1)
        return [results]


def mm_yolox_head_only(device='cuda:0'):
    config = 'C:\Work\split_computing\configs\yolox\yolox_s_8xb8-300e_coco.py'  # zmenit!!!!
    checkpoint = 'C:\Work\split_computing\checkpoints\yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'   # zmenit!!!
    model = mm_yolox_model(config, checkpoint, device=device)
    head = model
    tail = Identity()
    return head, tail


if __name__ == '__main__':
    import cv2
    input_img = cv2.imread('images/sample.jpg')

    head, tail = mm_yolox_head_only()

    output = head(input_img)
    output2 = tail(output)
