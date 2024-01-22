from mmdet.apis.det_inferencer_cust import DetInferencer_CUST
from mmdet.apis.det_inferencer import DetInferencer
import torch.nn as nn

img_path = '/home/toofy/PycharmProjects/mmdetection/demo/cat_demo.jpg'
config_path = '/home/toofy/PycharmProjects/mmdetection/configs/yolox/yolox_l_8xb8-300e_coco.py'
weights = '/home/toofy/PycharmProjects/mmdetection/configs/yolox/yolox_l_8x8_300e_coco.pth'
device = 'cpu'
out_dir = '/home/toofy/PycharmProjects/mmdetection/demo/outputs'
config_path_cust = '/home/toofy/PycharmProjects/mmdetection/configs/yolox/yolox_l_8xb8-300e_coco_custom.py'



def print_dims(tuple_out, type):
    for idx, level_out in enumerate(tuple_out):
        print(f"{type}, level: {idx} size: {tuple(level_out.shape)}")

def device_loader(split_type):
    yolox_l = DetInferencer(model=config_path, weights=weights, device=device)
    if split_type == "backbone":
        del (yolox_l.model.bbox_head)
        del (yolox_l.model.neck)
    elif split_type == "neck":
        del(yolox_l.model.bbox_head)

    return yolox_l

def device_predictor(yolox_device, inputs, split_type):
    if split_type == "backbone":
        return yolox_device.model.backbone(inputs)
    else:
        return yolox_device.model.neck(yolox_device.model.backbone(inputs))

def server_loader(split_type):
    yolox_l = DetInferencer(model=config_path, weights=weights, device=device)
    if split_type == "backbone":
        del (yolox_l.model.backbone)
    elif split_type == "neck":
        del(yolox_l.model.backbone)
        del(yolox_l.model.neck)

    return yolox_l

def server_predictor(yolox_server, inputs, split_type):
    if split_type == "backbone":
        return yolox_server.model.bbox_head(yolox_server.model.neck(inputs))
    else:
        return yolox_server.model.bbox_head(inputs)


def enc_256(inputs):
    conv = nn.Conv2d(256, 256, 3, 2, 1)
    bn = nn.BatchNorm2d(256)
    act = nn.SiLU()

    return act(bn(conv(inputs)))

def main():
    yolox_l = DetInferencer_CUST(model=config_path_cust, weights=weights, device=device)

    # yolox_l(img_path, out_dir=out_dir)


    # input = torch.rand(1, 3, 640, 640)
    #
    # res1 = yolox_l.model.backbone(input)
    # res2 = yolox_l.model.encoder(res1)
    # res3 = yolox_l.model.decoder(res2)
    # res4 = yolox_l.model.neck(res3)
    # res5 = yolox_l.model.bbox_head(res4)
    #
    # print_dims(res1, "backbone")
    # print_dims(res2, "encoder")
    # print_dims(res3, "decoder")
    # print_dims(res4, "neck")

    # train
    train_detector(config_path, weights, device=device, validate=True)

if __name__ == '__main__':
    main()