import torch
from ultralytics import YOLO

from models.utils import Identity


def ut_yolo8_tail_only():
    head = Identity()
    tail = YOLO("yolov8n.yaml")
    return head, tail


if __name__ == '__main__':
    head, tail = ut_yolo8_tail_only()
    rand_input = torch.rand((1, 3, 640, 640))
    output = head(rand_input)
    output2 = tail(output)
