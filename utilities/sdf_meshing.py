"""From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
"""
#!/usr/bin/env python3

import logging
import numpy as np
import plyfile
import skimage.measure
import time, os
import torch
from tqdm import tqdm


def create_mesh(
    decoder,
    filename,
    N=256,
    max_batch=32**3,
    offset=None,
    scale=None,
    output_return=True,
    verbose=True,
    manual_offset=None,
):
    ply_filename = filename

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    box_side_len = 2.0
    voxel_origin = torch.LongTensor([-box_side_len/2, -box_side_len/2, -box_side_len/2])
    voxel_size = box_side_len / (N - 1)

    num_samples = N**3

    overall_index = torch.arange(0, num_samples, 1, out=torch.LongTensor())
    samples = torch.zeros(num_samples, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:,:3] = samples[:,:3] * voxel_size + voxel_origin
    samples.requires_grad = False

    head = 0

    decoder.eval()
    pbar = tqdm(total=num_samples, desc="Computing SDF grid")
    while head < num_samples:
        if verbose:
            print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        output = decoder(sample_subset)

        samples[head : min(head + max_batch, num_samples), 3] = (
            output["model_out"].squeeze().detach().cpu()  # .squeeze(1)
        )
        head += max_batch
        pbar.update(max_batch)

        del output["model_out"]
        del output["model_in"]
        torch.cuda.empty_cache()
    pbar.close()

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    ply_filename_out = ply_filename+".ply" if ply_filename is not None else None
    mesh_points = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename_out,
        offset,
        manual_offset,
        scale,
        output_return=output_return,
    )

    if output_return:
        return mesh_points


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    manual_offset=None,
    scale=None,
    output_return=False,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = (
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros(0),
    )

    min = torch.amin(pytorch_3d_sdf_tensor.reshape(-1))
    max = torch.amax(pytorch_3d_sdf_tensor.reshape(-1))

    if not min * max > 0:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    else:
        print(f" nothing appeared min sdf: {min}, max sdf : {max}")

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if manual_offset is not None:
        print("compensated")
        mesh_points = mesh_points + manual_offset
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file
    if ply_filename_out is not None:
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]

        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        if ply_filename_out != ".ply":
            ply_data = plyfile.PlyData([el_verts, el_faces])
            logging.debug("saving mesh to %s" % (ply_filename_out))
            ply_data.write(ply_filename_out)
            print(os.getcwd(), ply_filename_out, "saved")
            logging.debug(
                "converting to ply format and writing to file took {} s".format(
                    time.time() - start_time
                )
            )

    return mesh_points


def write_files(output_file, pointcloud):
    with open(output_file, "a") as f:
        print(pointcloud.shape[0])
        for iidx in range(pointcloud.shape[0]):
            content = " ".join(str(x) for x in pointcloud[iidx, :]) + "\n"
            f.write(content)
