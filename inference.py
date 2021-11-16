from glob import glob
from pathlib import Path

import numpy as np
import torch
import torchvision
from numpy import sqrt
from PIL import Image
from torch.utils import data
from tqdm import tqdm

from args import get_args
from modules import network, resnet, transform


class Images(data.Dataset):
    def __init__(self, root, transform):
        extensions = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
        self.imgs = sum([glob(root + "/*" + ext) for ext in extensions], [])
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        file = self.imgs[idx]
        return file, self.transform(Image.open(file).convert("RGB"))


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Images(
        root=args.input_dir,
        transform=transform.Transforms(size=args.image_size).test_transform,
    )
    class_num = args.n_clusters

    data_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model_fp = args.model_path
    model.load_state_dict(torch.load(model_fp, map_location=device.type)["net"])
    model.to(device)

    with torch.inference_mode():
        model.eval()
        labels, files, imgs = [], [], []
        for step, (f, x) in enumerate(tqdm(data_loader)):
            x = x.to(device)
            c = model.forward_cluster(x)
            labels.append(c.cpu())
            files.append(f)
            imgs.append(x.cpu())
    labels = torch.cat(labels)
    files = np.concatenate(files)
    imgs = torch.cat(imgs)

    print("Number of clusters found:", len(np.unique(labels)))
    sizes = np.array([(labels == l).sum() for l in range(len(np.unique(labels)))])
    print(
        "Cluster sizes (min, 25%, median, 75%, max):",
        [np.min(sizes), np.percentile(sizes, 25), np.median(sizes), np.percentile(sizes, 75), np.max(sizes)],
    )
    for l in tqdm(np.unique(labels)):
        indices = labels == l
        cluster = imgs[indices]
        cluster = cluster[np.random.permutation(len(cluster))]
        cluster = cluster[:72]
        name = f"clusters/{Path(args.input_dir).stem}_label{l}_size{indices.sum()}"
        torchvision.utils.save_image(cluster, f"{name}.jpg", nrow=int(16 / 11 * sqrt(len(cluster))))
        with open(f"{name}.txt", "w") as file:
            file.write("\n".join(files[indices]))
