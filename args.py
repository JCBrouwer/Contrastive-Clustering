import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--input_dir", required=True)
    parser.add_argument(f"--n_clusters", default=200, type=int)
    parser.add_argument(f"--seed", default=42, type=int)
    parser.add_argument(f"--workers", default=8, type=int)
    parser.add_argument(f"--batch_size", default=256, type=int)
    parser.add_argument(f"--image_size", default=224, type=int)
    parser.add_argument(f"--epochs", default=1000, type=int)
    parser.add_argument(f"--resnet", default="ResNet34", choices=["ResNet18", "ResNet34", "ResNet50"], type=str)
    parser.add_argument(f"--feature_dim", default=128, type=int)
    parser.add_argument(f"--model_dir", default="modelzoo/", type=str)
    parser.add_argument(f"--model_path", type=str)
    parser.add_argument(f"--learning_rate", default=3e-4, type=float)
    parser.add_argument(f"--weight_decay", default=0.0, type=float)
    parser.add_argument(f"--instance_temperature", default=0.5, type=float)
    parser.add_argument(f"--cluster_temperature", default=1.0, type=float)
    return parser.parse_args()
