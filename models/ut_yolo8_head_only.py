import torch
from ultralytics import YOLO

from models.utils import Identity


def ut_yolo8_head_only():
    head = YOLO("yolov8n.yaml")
    tail = Identity()
    return head, tail


if __name__ == '__main__':
    head, tail = ut_yolo8_head_only()
    rand_input = torch.rand((1, 3, 640, 640))
    output = head(rand_input)
    output2 = tail(output)
