from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
from torch import Tensor
from torch.nn import Module
from typing import Union, List
from yolo8_modules import SplitDetectionPredictor, SplitDetectionModel


def predict_head(model: Module, input_image: Union[str, Tensor], save_layers: List[int], image_size: int = 640) -> dict:
    predictor = SplitDetectionPredictor(model, overrides={"imgsz": image_size})

    # Prepare data
    predictor.setup_source(input_image)
    batch = next(iter(predictor.dataset))
    predictor.batch = batch
    path, input_image, _, _ = batch

    # Preprocess
    preprocessed_image = predictor.preprocess(input_image)

    # Head predict
    y_head = model.forward_head(preprocessed_image, save_layers)
    y_head["img_shape"] = preprocessed_image.shape[2:]
    y_head["orig_shape"] = input_image[0].shape[:2]

    return y_head


def predict_tail(model: Module, y_head: dict, image_size: int = 640) -> Results:
    predictor = SplitDetectionPredictor(model, overrides={"imgsz": image_size})

    # Tail predict
    predictions = model.forward_tail(y_head)

    # Postprocess
    yolo_results = predictor.postprocess(predictions, y_head["img_shape"], y_head["orig_shape"])[0]

    return yolo_results


if __name__ == "__main__":
    splits = {
        "a": (10, [4, 6, 9]),
        "b": (16, [9, 12, 15]),
        "c": (22, [15, 18, 21])
    }
    split_layer, save_layers = splits["a"]

    # image = "../assets/coco_train_2017_small/000000005802.jpg"
    image = torch.rand(1, 3, 640, 640)

    # head prediction
    pretrained_model = YOLO("yolov8m.pt").model
    model_head = SplitDetectionModel(pretrained_model, split_layer=split_layer)
    model_head = model_head.eval()
    head_output = predict_head(model_head, image, save_layers)

    # tail prediction
    pretrained_model = YOLO("yolov8m.pt").model
    model_tail = SplitDetectionModel(pretrained_model, split_layer=split_layer)
    model_tail = model_tail.eval()
    results = predict_tail(model_tail, head_output)
    print(results.boxes)
    print(f"head layers: {len(model_head.head)}\ntail layers: {len(model_tail.tail)}")
