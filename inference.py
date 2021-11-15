import os

import numpy as np
import torch
from torch.utils import data

from args import get_args
from modules import network, resnet, transform
from train import Images

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Images(
        root=args.input_dir,
        transform=transform.Transforms(s=0.5, size=args.image_size),
    )
    class_num = args.n_clusters

    data_loader = data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)["net"])
    model.to(device)

    print("### Creating features from model ###")
    with torch.inference_mode():
        model.eval()
        feature_vector = []
        for step, x in enumerate(data_loader):
            x = x.to(device)
            c = model.forward_cluster(x)
            feature_vector.extend(c.cpu().numpy())
            if step % 20 == 0:
                print(f"Step [{step}/{len(data_loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
