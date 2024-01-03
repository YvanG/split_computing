import torch
from ultralytics import YOLO

from models.utils import Identity


class ut_yolo_model:
    def __init__(self, config):
        self.model = YOLO(config)

    def __call__(self, x):
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()/255   # v budoucnu lze napsat normalizaci
        x = self.model(x)
        boxes = x[0].boxes.data.cpu().numpy()
        return [boxes]


def ut_yolo8_head_only():
    head = ut_yolo_model(config="yolov8n.pt")
    tail = Identity()
    return head, tail


if __name__ == '__main__':
    import cv2
    input_img = cv2.imread('images/sample.jpg')

    head, tail = ut_yolo8_head_only()

    output = head(input_img)
    output2 = tail(output)
