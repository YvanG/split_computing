from ultralytics import YOLO
from yolo8_modules import SplitDetectionPredictor, SplitDetectionModel
import torch


def predict_head(model, im0s, save_layers, imgsz=640):
    predictor = SplitDetectionPredictor(model, overrides={"imgsz": imgsz})

    # Prepare data
    predictor.setup_source(im0s)
    batch = next(iter(predictor.dataset))
    predictor.batch = batch
    path, im0s, vid_cap, s = batch

    # Preprocess
    im = predictor.preprocess(im0s)

    # Head predict
    y_head = model.forward_head(im, save_layers)
    y_head["img_shape"] = im.shape[2:]
    y_head["orig_shape"] = im0s[0].shape[:2]

    return y_head


def predict_tail(model, y_head, imgsz=640):
    predictor = SplitDetectionPredictor(model, overrides={"imgsz": imgsz})

    # Tail predict
    preds = model.forward_tail(y_head)

    # Postprocess
    results = predictor.postprocess(preds, y_head["img_shape"], y_head["orig_shape"])

    return results


if __name__ == "__main__":
    splits = {
        "a": (10, [4, 6, 9]),
        "b": (16, [9, 12, 15]),
        "c": (22, [15, 18, 21])
    }
    split_layer, save_layers = splits["a"]

    image_path = "../assets/coco_train_2017_small/000000005802.jpg"
    image = torch.rand(1, 3, 640, 640)

    pretrained_model = YOLO("yolov8m.pt").model
    model_head = SplitDetectionModel(pretrained_model, split_layer=split_layer)
    model_head = model_head.eval()
    head_output = predict_head(model_head, image_path, save_layers)

    pretrained_model = YOLO("yolov8m.pt").model
    model_tail = SplitDetectionModel(pretrained_model, split_layer=split_layer)
    model_tail = model_tail.eval()
    results = predict_tail(model_tail, head_output)
    print(results[0].boxes)
    print(f"head layers: {len(model_head.head)}\ntail layers: {len(model_tail.tail)}")
