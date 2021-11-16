# fmt: off
import argparse
import multiprocessing

feature_nets = [
    "ResNet18", "ResNet34", "ResNet50",
    "CLIP-RN50", "CLIP-RN101", "CLIP-RN50x4", "CLIP-RN50x16", "CLIP-ViT-B/32", "CLIP-ViT-B/16",
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--input_dir", required=True)
    parser.add_argument(f"--n_clusters", default=200, type=int)
    parser.add_argument(f"--seed", default=42, type=int)
    parser.add_argument(f"--batch_size", default=512, type=int)
    parser.add_argument(f"--workers", default=1, type=int)
    parser.add_argument(f"--extract_batch_size", default=64, type=int)
    parser.add_argument(f"--extract_workers", default=multiprocessing.cpu_count(), type=int)
    parser.add_argument(f"--image_size", default=224, type=int)
    parser.add_argument(f"--epochs", default=100, type=int)
    parser.add_argument(f"--feature_net", default="CLIP-ViT-B/32", choices=feature_nets)
    parser.add_argument(f"--feature_dim", default=128, type=int)
    parser.add_argument(f"--model_dir", default="modelzoo/", type=str)
    parser.add_argument(f"--model_path", type=str)
    parser.add_argument(f"--learning_rate", default=3e-4, type=float)
    parser.add_argument(f"--weight_decay", default=0.0, type=float)
    parser.add_argument(f"--instance_temperature", default=0.5, type=float)
    parser.add_argument(f"--cluster_temperature", default=1.0, type=float)
    return parser.parse_args()
