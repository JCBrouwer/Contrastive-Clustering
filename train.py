import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils import data

from args import get_args
from modules import contrastive_loss, network, resnet, transform


class Images(data.Dataset):
    def __init__(self, root, transform):
        extensions = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
        self.imgs = sum([glob(root + "/*" + ext) for ext in extensions], [])
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.imgs[idx]).convert("RGB"))


def save_model(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_dir, f"{Path(args.input_dir).stem}_{current_epoch}.tar")
    state = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": current_epoch}
    torch.save(state, out)


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = Images(
        root=args.input_dir,
        transform=transform.Transforms(s=0.5, size=args.image_size),
    )
    class_num = args.n_clusters
    data_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to("cuda")
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device
    )
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    # train
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = 0
        for step, (x_i, x_j) in enumerate(data_loader):
            optimizer.zero_grad()
            x_i = x_i.to("cuda")
            x_j = x_j.to("cuda")
            z_i, z_j, c_i, c_j = model(x_i, x_j)
            loss_instance = criterion_instance(z_i, z_j)
            loss_cluster = criterion_cluster(c_i, c_j)
            loss = loss_instance + loss_cluster
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(
                    f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}"
                )
            loss_epoch += loss.item()
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)
