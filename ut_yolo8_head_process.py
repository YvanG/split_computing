from models.ut_yolo8_head_only import ut_yolo8_head_only

class YoloProcess:

    def __init__(self):
        self.head, self.tail = ut_yolo8_head_only()

    def initialize(self):
        print("Hello from UT_YOLO_HEAD_PROCESS")
        pass

    def process_head(self, data=None):
        output = self.head(data)
        return output

    def process_tail(self, data=None):
        output = self.tail(data)
        return output

