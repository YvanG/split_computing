from mmdet.apis import DetInferencer
import torch

config_path = '../configs/yolox/yolox_l_8xb8-300e_coco.py'
weights = '../yolox_weights/yolox_l_8x8_300e_coco.pth'
device = 'cpu'

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
def main():
    split_type = "backbone"
    inputs = torch.rand(1, 3, 640, 640)
    yolox_device = device_loader(split_type)
    device_out = device_predictor(yolox_device, inputs, split_type)

    yolox_server = server_loader(split_type)
    server_res = server_predictor(yolox_server, device_out, split_type)

    print_dims(device_out, "device_out")
    print(server_res) # TODO print_dims nefunguje u serveru

    # train
    # train_detector(config_path, weights, device=device, validate=True)

if __name__ == '__main__':
    main()