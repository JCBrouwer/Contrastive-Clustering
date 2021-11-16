from pathlib import Path

import numpy as np
import torch
import torchvision
from numpy import sqrt
from tqdm import tqdm

from args import get_args
from data import DataPrefetcher, Images, Transforms
from extractors import get_feature_network
from projector import Projector

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method("spawn")
    args = get_args()

    transform = Transforms(args.image_size).test_transform
    dataset = Images(root=args.input_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    data_prefetcher = DataPrefetcher(data_loader)

    feature_extractor = get_feature_network(args.feature_net).eval().float().to(device)

    class_num = args.n_clusters
    model = Projector(feature_extractor.rep_dim, args.feature_dim, class_num).eval().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device.type)["net"])

    with torch.inference_mode(), tqdm(total=len(dataset), smoothing=0) as pbar:
        labels, files, imgs = [], [], []

        f, x = data_prefetcher.next()
        while x is not None:
            h = feature_extractor(x)
            c = model.forward_cluster(h)

            imgs.append(x.cpu())
            labels.append(c.cpu())
            files.append(f)

            pbar.update(len(x))
            f, x = data_prefetcher.next()

        labels, files, imgs = torch.cat(labels), np.concatenate(files), torch.cat(imgs)

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
