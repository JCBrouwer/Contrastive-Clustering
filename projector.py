import torch
import torch.nn as nn
from torch.nn.functional import normalize


class Projector(nn.Module):
    def __init__(self, rep_dim, feature_dim, class_num):
        super(Projector, self).__init__()
        self.instance_projector = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.ReLU(),
            nn.Linear(rep_dim, feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.ReLU(),
            nn.Linear(rep_dim, class_num),
            nn.Softmax(dim=1),
        )

    def forward(self, h_i, h_j):
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, h):
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
