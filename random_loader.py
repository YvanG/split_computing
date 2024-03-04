import numpy as np

class RandomLoader:
    
    def __init__(self):
        self.datacount = 1
        pass
    
    def initialize(self, options=None):
        print("Hello from SCLoader")
        pass    
    
    def next(self):
        if self.datacount > 0:
            output = np.round(np.random.rand(640, 640, 3)*255).astype(np.uint8)
            self.datacount -= 1
            return output
        else:
            return False
    
