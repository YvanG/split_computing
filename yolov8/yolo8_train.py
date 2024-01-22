import argparse
import torch
from collections import OrderedDict
from functools import partial
from yolo8_modules import YOLO2

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--model_name',
                        default='../configs/yolo8/models/yolov8m_a16_bottleneck.yaml',
                        type=str)
    parser.add_argument('--data_path',
                        default="../configs/yolo8/datasets/coco128.yaml", type=str)
    parser.add_argument('--project', default="", type=str)
    parser.add_argument('--name', default="", type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--optimizer', default="AdamW", type=str)
    parser.add_argument('--lr0', default=0.001, type=float)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--imgsz', default=640, type=int)
    parser.add_argument('--cos_lr', default=False, action='store_true')
    parser.add_argument('--save_period', default=-1, type=int)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--yolo_checkpoint', default="", type=str)
    parser.add_argument('--freez', default=False, action='store_true')

    return parser


def freeze_layer(trainer, train_layer_names=(), prefix="model"):
    model = trainer.model
    num_freeze = 0
    if prefix:
        train_layer_names = [f"{prefix}.{n}" for n in train_layer_names]
    print(f"Freezing {train_layer_names} layers")

    for k, v in model.named_parameters():
        if k not in train_layer_names:
            #     v.requires_grad = True
            # else:
            print(f'freezing {k}')
            num_freeze += 1
            v.requires_grad = False

    print(f"{num_freeze} layers are frozen.")


def load_parameters(model, yolo_path, prefix="model"):
    model = model.model
    _model = model
    model = model.model.state_dict()

    weights = torch.load(yolo_path)["model"].model.state_dict()

    assigned_names = []
    renamed_weights = OrderedDict()
    for name in weights:
        _name = name.split(".")
        layer_name = ".".join(_name[1:])
        layer_shape = weights[name].shape

        for name_model in model:
            if name_model in assigned_names:
                continue
            _name_model = name_model.split(".")
            layer_name_model = ".".join(_name_model[1:])
            layer_shape_model = model[name_model].shape

            if layer_name == layer_name_model and layer_shape == layer_shape_model:
                assigned_names.append(name_model)
                name_model = f"{prefix}.{name_model}"
                renamed_weights[name_model] = weights[name].clone()
                break

    missing_names = []
    for name_model in model:
        if name_model not in assigned_names:
            missing_names.append(name_model)

    _model.load_state_dict(renamed_weights, strict=False)
    return missing_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)

    model_args = {
        "data": args.data_path,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "lr0": args.lr0,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "cos_lr": args.cos_lr,
        "save_period": args.save_period,
        "seed": args.seed,
    }

    model = YOLO2(args.model_name)
    if args.yolo_checkpoint:
        model_0 = YOLO2(args.yolo_checkpoint)
        missing_names = load_parameters(model, args.yolo_checkpoint)
        model.overrides = model_0.overrides
        model.ckpt_path = model_0.ckpt_path
        model.cfg = model_0.cfg
        model.ckpt = model_0.ckpt
        model.ckpt["model"] = model.model

    if args.freez:
        model.add_callback("on_train_start", partial(freeze_layer, train_layer_names=missing_names))

    model.train(
        **model_args
    )
