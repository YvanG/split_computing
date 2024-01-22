# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage_cust import SingleStageDetector_CUST
# from mmdet.apis.det_inferencer_cust import DetInferencer_CUST
import torch.nn as nn
from mmengine.model import BaseModule

@MODELS.register_module()
class YOLOX_SPLIT_Encoder(BaseModule):
    def __init__(self,
                 in_channels: list,
                 out_channels: list,
                 kernel_size: int,
                 stride: int,
                 padding: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(in_channels[0])
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(in_channels[1])
        self.act2 = nn.SiLU()

        self.conv3 = nn.Conv2d(in_channels[2], out_channels[2], kernel_size, stride, padding)
        self.bn3 = nn.BatchNorm2d(in_channels[2])
        self.act3 = nn.SiLU()

    def forward(self, x):
        x1 = self.act1(self.bn1(self.conv1(x[0])))
        x2 = self.act2(self.bn2(self.conv2(x[1])))
        x3 = self.act3(self.bn3(self.conv3(x[2])))
        return (x1, x2, x3)


@MODELS.register_module()
class YOLOX_SPLIT_Decoder(nn.Module):
    def __init__(self,
                 in_channels: list,
                 out_channels: list,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_padding: int):
        super(YOLOX_SPLIT_Decoder, self).__init__()

        self.conv_transpose1 = nn.ConvTranspose2d(in_channels[0], out_channels[0], kernel_size, stride, padding, output_padding)
        self.bn1 = nn.BatchNorm2d(in_channels[0])
        self.act1 = nn.SiLU()

        self.conv_transpose2 = nn.ConvTranspose2d(in_channels[1], out_channels[1], kernel_size, stride, padding, output_padding)
        self.bn2 = nn.BatchNorm2d(in_channels[1])
        self.act2 = nn.SiLU()

        self.conv_transpose3 = nn.ConvTranspose2d(in_channels[2], out_channels[2], kernel_size, stride, padding, output_padding)
        self.bn3 = nn.BatchNorm2d(in_channels[2])
        self.act3 = nn.SiLU()

    def forward(self, x):
        x1 = self.act1(self.bn1(self.conv_transpose1(x[0])))
        x2 = self.act2(self.bn2(self.conv_transpose2(x[1])))
        x3 = self.act3(self.bn3(self.conv_transpose3(x[2])))
        return (x1, x2, x3)

@MODELS.register_module()
class YOLOX_CUST(SingleStageDetector_CUST):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLOX. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLOX. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 encoder: ConfigType,
                 decoder: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
