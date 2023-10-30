import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from .sdf_meshing import create_mesh
from pytorch3d.loss import chamfer_distance


def data2pairs(data):
    pairs = {}
    for shape_idx in data:
        pairs[shape_idx] = []
        for deform_idx in data[shape_idx]:
            if deform_idx != "nominal":
                pairs[shape_idx].append(deform_idx)
    return pairs


def pcd_interest(points):
    points = points.squeeze()
    inlier_idx = np.where((points[:, 2] < 1) & (points[:, 2] > -1))[0]
    return points[inlier_idx, :]


def validation_3d(
    dat, decoder, filename="", scale=1, manual_offset=None, offset=[0, 0, 0], eps=0.0
):
    n = 5600

    source_cloud = create_mesh(
        decoder,
        filename,
        N=256,
        output_return=True,
        verbose=False,
        scale=scale,
        offset=offset,
        manual_offset=manual_offset,
    )

    source_cloud = pcd_interest(source_cloud)
    iidx = np.random.permutation(source_cloud.shape[0])[:n]
    source_cloud = torch.tensor(source_cloud[iidx, :]).unsqueeze(0).float()

    def_on_surf_idx = torch.where(torch.abs(dat["gt"]) <= eps)[1]
    target_cloud = dat["coords"].reshape((-1, 3))[def_on_surf_idx, :]
    target_cloud = pcd_interest(target_cloud).unsqueeze(0)
    iidx = np.random.permutation(target_cloud.shape[1])[:n]
    target_cloud = target_cloud[:, iidx, :]

    dist_bidirectional, _ = chamfer_distance(
        source_cloud.float(), target_cloud.float()
    )  # source, target

    # fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection="3d"), figsize=(10, 5))
    #
    # ax[0].scatter(source_cloud[:, :, 0], source_cloud[:, :, 1], source_cloud[:, :, 2], s=0.04)
    # ax[0].scatter(target_cloud[:, :, 0], target_cloud[:, :, 1], target_cloud[:, :, 2], s=0.03, c='red')
    #
    # ax[0].set_xlim3d(-0.6, 0.6)
    # ax[0].set_ylim3d(-0.5, 0.5)
    # ax[0].set_zlim3d(-1, 1)
    # ax[0].grid()
    # ax[0].view_init(-5, 90)
    #
    # ax[1].scatter(source_cloud[:, :, 0], source_cloud[:, :, 1], source_cloud[:, :, 2], s=0.04)
    # ax[1].scatter(target_cloud[:, :, 0], target_cloud[:, :, 1], target_cloud[:, :, 2], s=0.03, c='red')
    #
    # ax[1].set_xlim3d(-1, 1)
    # ax[1].set_ylim3d(-1, 1)
    # ax[1].set_zlim3d(-1, 1)
    # ax[1].grid()
    # ax[1].view_init(-5, 10)
    #
    # plt.title(dist_bidirectional)
    # plt.show()

    return dist_bidirectional


def make_dir(path):
    # Make dir if not exist
    if not os.path.exists(path):
        os.makedirs(path)
