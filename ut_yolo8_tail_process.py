from models.ut_yolo8_tail_only import ut_yolo8_tail_only

class YoloProcess:

    def __init__(self):
        self.head, self.tail = ut_yolo8_tail_only()

    def initialize(self):
        print("Hello from UT_YOLO_TAIL_PROCESS")
        pass

    def process_head(self, data=None):
        output = self.head(data)
        return output

    def process_tail(self, data=None):
        output = self.tail(data)
        return output
