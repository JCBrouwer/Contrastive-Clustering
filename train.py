import os
from pathlib import Path

import torch
from tqdm import tqdm

from args import get_args
from contrastive_loss import ClusterLoss, InstanceLoss
from data import Features, extract_features
from extractors import get_feature_network
from projector import Projector


def save_model(args, model, optimizer, feature_net, current_epoch):
    out = os.path.join(args.model_dir, f"{Path(args.input_dir).stem}_{feature_net}_{current_epoch}.tar")
    state = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": current_epoch}
    torch.save(state, out)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method("spawn")
    args = get_args()
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # pre-extract features for every image
    features_file = f"cache/{Path(args.input_dir).stem}_features{args.feature_net.replace('/', '_')}.npy"
    if not os.path.exists(features_file):
        rep_dim = extract_features(
            args.input_dir,
            features_file,
            args.feature_net,
            args.image_size,
            args.extract_batch_size,
            args.extract_workers,
        )
    else:
        rep_dim = get_feature_network(args.feature_net).rep_dim

    data_loader = torch.utils.data.DataLoader(
        Features(features_file),
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=4,
        shuffle=True,
        drop_last=True,
    )

    # initialize model, optimizer, and losses
    class_num = args.n_clusters
    model = Projector(rep_dim, args.feature_dim, class_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion_instance = InstanceLoss(args.batch_size, args.instance_temperature).to(device)
    criterion_cluster = ClusterLoss(class_num, args.cluster_temperature).to(device)

    # resume training from a checkpoint
    start_epoch = 0
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    with tqdm(range(start_epoch, args.epochs)) as pbar:
        for epoch in pbar:

            loss_epoch = torch.zeros([], device=device)
            loss_instance_epoch = torch.zeros([], device=device)
            loss_cluster_epoch = torch.zeros([], device=device)
            for step, (x_i, x_j) in enumerate(data_loader):

                optimizer.zero_grad()
                x_i = x_i.to(device)
                x_j = x_j.to(device)
                z_i, z_j, c_i, c_j = model(x_i, x_j)
                loss_instance = criterion_instance(z_i, z_j)
                loss_cluster = criterion_cluster(c_i, c_j)
                loss = loss_instance + loss_cluster
                loss.backward()
                optimizer.step()

                loss_epoch += loss.detach()
                loss_instance_epoch += loss_instance.detach()
                loss_cluster_epoch += loss_cluster.detach()

            if epoch % 10 == 0:
                save_model(args, model, optimizer, args.feature_net.replace("/", "_"), epoch)

            pbar.write(
                f"Epoch [{epoch}/{args.epochs}]".ljust(20)
                + f"Total: {loss_epoch.item() / len(data_loader):.4f}".ljust(20)
                + f"Instance: {loss_instance_epoch.item() / len(data_loader):.4f}".ljust(20)
                + f"Cluster: {loss_cluster_epoch.item() / len(data_loader):.4f}"
            )

    save_model(args, model, optimizer, args.feature_net.replace("/", "_"), args.epochs)
