from models.ut_yolo8_head_only import ut_yolo8_head_only
from models.ut_yolo8_tail_only import ut_yolo8_tail_only
from models.mm_yolox_head_only import mm_yolox_head_only
from models.mm_yolox_tail_only import mm_yolox_tail_only


class SCProcess:

    def __init__(self, model_name, device='cuda:0'):
        if model_name == 'ut_yolo8_head_only':
            self.head, self.tail = ut_yolo8_head_only()
        elif model_name == 'ut_yolo8_tail_only':
            self.head, self.tail = ut_yolo8_tail_only()
        elif model_name == 'mm_yolox_head_only':
            self.head, self.tail = mm_yolox_head_only(device=device)
        elif model_name == 'mm_yolox_tail_only':
            self.head, self.tail = mm_yolox_tail_only(device=device)

    def initialize(self):
        print("Hello from SCProcess")
        pass

    def process_head(self, data=None):
        output = self.head(data)
        return output

    def process_tail(self, data=None):
        output = self.tail(data)
        return output


if __name__ == '__main__':
    import torch

    model_name = 'ut_yolo8_head_only'
    rand_input = torch.rand((1, 3, 640, 640))

    sc = SCProcess(model_name=model_name)
    output_head = sc.process_head(data=rand_input)
    output_tail = sc.process_tail(data=output_head)
