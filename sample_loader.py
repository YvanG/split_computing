import numpy as np
import cv2

class SampleLoader:
    
    def __init__(self):
        self.datacount = 1
        pass
    
    def initialize(self, options=None):
        print("Hello from SCLoader")
        pass    
    
    def next(self):
        if self.datacount > 0:
            output = cv2.imread("./sample.jpg")
            self.datacount -= 1
            return output
        else:
            return False
    
