from typing import Union, List

import numpy as np
from torch import Tensor
from torch.nn import Module
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
from yolo8_modules import SplitDetectionPredictor, SplitDetectionModel

class ut_yolo_model:
    def __init__(self, config, split_layer):
        torch.cuda.set_device(0)
        pretrained_model = YOLO(config).model
        model_head = SplitDetectionModel(pretrained_model, split_layer=split_layer)
        model_head = model_head.eval()
        model_head.to("cuda:0")
        self.model = model_head

    def __call__(self, x):
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()/255   # v budoucnu lze napsat normalizaci
        x = self.model(x)
        boxes = x[0].boxes.data.cpu().numpy()
        return [boxes]

def ut_yolo8(config, split_layer):
    head = ut_yolo_model(config=config, split_layer=split_layer)
    tail = ut_yolo_model(config=config, split_layer=split_layer)
    return head, tail

class YoloProcess:

    def __init__(self):
        split = "c"
        self.splits = {
            "a": (10, [4, 6, 9]),
            "b": (16, [9, 12, 15]),
            "c": (22, [15, 18, 21])
        }        
        
        bottleneck_splits = {
            "a": (13, [10, 11, 12]),
            "b": (19, [16, 17, 18]),
            "c": (25, [22, 23, 24])
        }
        
        #self.split_layer, self.save_layers =  self.splits[split]
        self.split_layer, self.save_layers =  bottleneck_splits[split]
        
        self.head, self.tail = ut_yolo8(config="/path_to_weights/last.pt", split_layer=self.split_layer)
        self.head_predictor = SplitDetectionPredictor(self.head.model, overrides={"imgsz": 640})
        self.tail_predictor = SplitDetectionPredictor(self.tail.model, overrides={"imgsz": 640})

    def initialize(self):
        print("Hello from UT_YOLO_PROCESS")
        pass
    
    def predict_head(self, input_image: Union[str, Tensor], save_layers: List[int], image_size: int = 640) -> dict:
        #predictor = SplitDetectionPredictor(self.head.model, overrides={"imgsz": image_size})

        # Prepare data
        self.head_predictor.setup_source(input_image)
        batch = next(iter(self.head_predictor.dataset))
        self.head_predictor.batch = batch
        path, input_image, _, _ = batch
        
        # Preprocess
        preprocessed_image = self.head_predictor.preprocess(input_image)
        if isinstance(input_image, list):
            input_image = np.array([np.moveaxis(img, -1, 0) for img in input_image])

        # Head predict
        y_head_dict = self.head.model.forward_head(preprocessed_image, save_layers)
        y_head = []
        
        layers_output = []
        
        #y_head.append(np.array(len(y_head_dict["layers_output"])))
        
        for layer in y_head_dict["layers_output"]:
            if layer == None:
                y_head.append(np.array([-1]))
            else:
                print(layer.dtype)
                y_head.append(layer.numpy())
        
        #y_head.append(layers_output)
        y_head.append(np.array([y_head_dict["last_layer_idx"]]))
        y_head.append(np.array(preprocessed_image.shape[2:]))
        y_head.append(np.array(input_image.shape[2:]))
        #y_head["img_shape"] = preprocessed_image.shape[2:]
        #y_head["orig_shape"] = input_image.shape[2:]

        return y_head

    def predict_tail(self, y_head: dict, image_size: int = 640) -> Results:
        #predictor = SplitDetectionPredictor(self.tail.model, overrides={"imgsz": image_size})

        # Tail predict
        predictions = self.tail.model.forward_tail(y_head)

        # Postprocess
        yolo_results = self.tail_predictor.postprocess(predictions, y_head["img_shape"], y_head["orig_shape"])[0]

        return yolo_results

    def process_head(self, data=None):      
        input_image = torch.from_numpy(data)
        input_image = torch.rand(1, 3, 640, 640)
        output = self.predict_head(input_image, self.save_layers)
        #output = SplitDetectionPredictor(self.head, data,)
        return output

    def process_tail(self, data=None):
        y_head = {}
        y_head["orig_shape"] = tuple(data.pop())
        y_head["img_shape"] = tuple(data.pop())
        y_head["last_layer_idx"] = data.pop()[0]
        
        layers_output = []
        
        for layer in data:
            if len(layer.shape) == 1 and layer.shape[0] == 1 and layer[0] == -1:
                layers_output.append(None)
            else:
                layers_output.append(torch.from_numpy(layer).type(torch.float32).to("cuda:0"))
        
        y_head["layers_output"] =layers_output
        
        output = self.predict_tail(y_head)
        return output

