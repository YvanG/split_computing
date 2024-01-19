import argparse
from typing import Union, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from ultralytics import YOLO
from ultralytics.engine.results import Results

from yolo8_modules import SplitDetectionPredictor, SplitDetectionModel


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--split', default='', type=str)
    parser.add_argument('--bottleneck', default=False, action='store_true')

    return parser


def predict_head(model: Module, input_image: Union[str, Tensor], save_layers: List[int], image_size: int = 640) -> dict:
    predictor = SplitDetectionPredictor(model, overrides={"imgsz": image_size})

    # Prepare data
    predictor.setup_source(input_image)
    batch = next(iter(predictor.dataset))
    predictor.batch = batch
    path, input_image, _, _ = batch

    # Preprocess
    preprocessed_image = predictor.preprocess(input_image)
    if isinstance(input_image, list):
        input_image = np.array([np.moveaxis(img, -1, 0) for img in input_image])

    # Head predict
    y_head = model.forward_head(preprocessed_image, save_layers)
    y_head["img_shape"] = preprocessed_image.shape[2:]
    y_head["orig_shape"] = input_image.shape[2:]

    return y_head


def predict_tail(model: Module, y_head: dict, image_size: int = 640) -> Results:
    predictor = SplitDetectionPredictor(model, overrides={"imgsz": image_size})

    # Tail predict
    predictions = model.forward_tail(y_head)

    # Postprocess
    yolo_results = predictor.postprocess(predictions, y_head["img_shape"], y_head["orig_shape"])[0]

    return yolo_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    # model
    default_splits = {
        "a": (10, [4, 6, 9]),
        "b": (16, [9, 12, 15]),
        "c": (22, [15, 18, 21])
    }
    # bottleneck model
    bottleneck_splits = {
        "a": (13, [10, 11, 12]),
        "b": (19, [16, 17, 18]),
        "c": (25, [22, 23, 24])
    }

    splits = None
    if args.bottleneck:
        splits = bottleneck_splits
    else:
        splits = default_splits
    split_layer, save_layers = splits[args.split]

    # input
    image = "../assets/coco_train_2017_small/000000005802.jpg"
    image = torch.rand(1, 3, 640, 640)

    # head prediction
    pretrained_model = YOLO(args.model_path).model
    model_head = SplitDetectionModel(pretrained_model, split_layer=split_layer)
    model_head = model_head.eval()
    head_output = predict_head(model_head, image, save_layers)

    # debug values
    shapes = [o.shape for o in head_output["layers_output"] if o is not None]

    # tail prediction
    pretrained_model = YOLO(args.model_path).model
    model_tail = SplitDetectionModel(pretrained_model, split_layer=split_layer)
    model_tail = model_tail.eval()
    results = predict_tail(model_tail, head_output)

    # print(results.boxes, "\n")
    print("head output shapes:")
    for o in shapes:
        print(o)

    print(f"\nhead layers: {len(model_head.head)}\ntail layers: {len(model_tail.tail)}")
