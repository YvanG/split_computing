from yolo8_modules import YOLO2
import argparse
import os


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--checkpoint_path', default=r'D:\MLProjects\split_computing\checkpoints\133a-bottleneck2_a64\weights\last.pt', type=str)
    parser.add_argument('--dataset_path', default=r'D:\MLProjects\split_computing\configs\yolo8\datasets\coco1.yaml', type=str, help='Path to yolo_dataset.yaml')

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    model = YOLO2(args.checkpoint_path)
    metrics = model.val(data=args.dataset_path, imgsz=640, batch=1, save_json=True, device="cpu")
    results = metrics.results_dict

    print(f"{os.path.dirname(args.checkpoint_path)}")
    print(f'mAP50-95(B): {results["metrics/mAP50-95(B)"]:<7.3f}\n'
          f'mAP50(B): {results["metrics/mAP50(B)"]:<7.3f}\n'
          f'mAP75(B): {results["metrics/mAP75(B)"]:<7.3f}')

