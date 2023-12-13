from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.nn import yaml_model_load, parse_model
from ultralytics.nn.modules import (Detect, Pose, Segment)
from ultralytics.utils import LOGGER
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import initialize_weights, model_info


class SplitDetectionModel(nn.Module):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True,
                 split_layer=-1):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, str):
            """Initialize the YOLOv8 detection model with the given config and parameters."""
            self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

            # Define model
            ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
            if nc and nc != self.yaml['nc']:
                LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
                self.yaml['nc'] = nc  # override YAML value

            self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
            self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
            self.inplace = self.yaml.get('inplace', True)

            # Build strides
            m = self.model[-1]  # Detect()
            if isinstance(m, (Detect, Segment, Pose)):
                s = 256  # 2x min stride
                m.inplace = self.inplace
                forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
                m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
                self.stride = m.stride
                m.bias_init()  # only run once
            else:
                self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

            # Init weights, biases
            initialize_weights(self)
            if verbose:
                self.info()
                LOGGER.info('')
            self.pt = True
        else:
            # load data from detection model
            self.model = cfg.model
            self.save = cfg.save
            self.stride = cfg.stride
            self.inplace = cfg.inplace
            self.names = cfg.names
            self.yaml = cfg.yaml
            self.nc = cfg.nc
            self.task = cfg.task
            self.pt = True

        if split_layer > 0:
            self.head = self.model[:split_layer]
            self.tail = self.model[split_layer:]

    def info(self, detailed=False, verbose=True, imgsz=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def forward_head(self, x, output_from=[]):
        y, dt = [], []  # outputs
        for i, m in enumerate(self.head):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            if (m.i in self.save) or (i in output_from):
                y.append(x)
            else:
                y.append(None)

        if y[i] is None:
            y[i] = x
        return {"layers_output": y, "last_layer_idx": i}

    def forward_tail(self, x):
        y, dt = [], []  # outputs
        y = x["layers_output"]
        x = y[x["last_layer_idx"]]
        for m in self.tail:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        y = x
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def _predict_once(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def forward(self, x):
        return self._predict_once(x)

    def from_numpy(self, x):
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x


class SplitDetectionPredictor(DetectionPredictor):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        model.fp16 = self.args.half
        self.model = model

    def pre_transform(self, im):
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt,
                              stride=max(int(self.model.stride.max()), 32))
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img_shape, orig_shape, orig_imgs=None):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if orig_imgs is not None and not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            if orig_imgs is None:
                orig_img = np.empty([0,0,0,0])
                img_path = ""
            else:
                orig_img = orig_imgs[i]
                img_path = self.batch[0][i]

            pred[:, :4] = ops.scale_boxes(img_shape, pred[:, :4], orig_shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results