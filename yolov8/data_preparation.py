import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dataset_root', default='', type=str)
    parser.add_argument('--pad_color', default=114, type=int)

    return parser


def resize(image, size=640):
    h, w = image.shape[:2]
    if h >= size or w >= size:
        return image

    scale = 640 / np.max([h, w])
    nw, nh = np.round(np.array([w, h]) * scale).astype(int)
    image = cv2.resize(image, (nw, nh))

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    images_path = os.path.join(args.dataset_root, "images")
    labels_path = os.path.join(args.dataset_root, "labels")
    folder_name = "val2017"

    new_folder = os.path.join(images_path, folder_name + "_padded")
    new_label_folder = os.path.join(labels_path, folder_name + "_padded")
    os.makedirs(new_folder, exist_ok=True)
    os.makedirs(new_label_folder, exist_ok=True)

    image_names = os.listdir(os.path.join(images_path, folder_name))
    for name in tqdm(image_names):
        image_path = os.path.join(images_path, folder_name, name)
        label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")

        image = cv2.imread(image_path)
        image = resize(image)
        h, w = image.shape[:2]

        size = np.max([h, w])
        image_padded = np.ones([size, size, 3]).astype(np.uint8)
        image_padded = image_padded * args.pad_color

        # add padding
        dif = np.abs(w - h)
        pad_value_0 = np.floor(dif / 2).astype(int)

        if w > h:
            pad_value_1 = pad_value_0 + h
            image_padded[pad_value_0:pad_value_1, :] = image
        else:
            pad_value_1 = pad_value_0 + w
            image_padded[:, pad_value_0:pad_value_1] = image

        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                new_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    line = line.split(" ")

                    cls, xc, yc, bw, bh = int(line[0]), float(line[1]) * w, float(line[2]) * h, float(
                        line[3]) * w, float(line[4]) * h
                    if w > h:
                        yc += pad_value_0
                    else:
                        xc += pad_value_0

                    xc, yc, bw, bh = xc / size, yc / size, bw / size, bh / size
                    new_lines.append(" ".join([str(i) for i in [cls, xc, yc, bw, bh]]))

            with open(os.path.join(new_folder, name).replace("images", "labels").replace(".jpg", ".txt"), "w") as f:
                new_lines = "\n".join(new_lines)
                f.write(new_lines)

        cv2.imwrite(os.path.join(new_folder, name), image_padded)
