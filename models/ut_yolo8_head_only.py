import torch
from ultralytics import YOLO

from models.utils import Identity


class ut_yolo_model:
    def __init__(self, config):
        self.model = YOLO(config)

    def __call__(self, x):
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()/255   # v budoucnu lze napsat normalizaci
        return self.model(x)


def ut_yolo8_head_only():
    head = ut_yolo_model(config="yolov8n.pt")
    tail = Identity()
    return head, tail


if __name__ == '__main__':
    import numpy as np

    rand_input = np.round(np.random.rand(640, 640, 3)*255).astype(np.uint8)     # simulate input from camera

    head, tail = ut_yolo8_head_only()

    output = head(rand_input)
    output2 = tail(output)
