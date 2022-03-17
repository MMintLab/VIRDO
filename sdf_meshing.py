'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3

import logging
import numpy as np
import plyfile
import skimage.measure
import time, os
import torch
import diff_operators


def create_mesh(
    decoder, filename, N=256, max_batch=40 ** 3, offset=None, scale=None, output_return = True, verbose = True
):
    start = time.time()
    ply_filename = filename



    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)
    normals = torch.zeros(N **3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0
    
    decoder.eval()
    while head < num_samples:
        if verbose:
            print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        output = decoder(sample_subset)

        samples[head : min(head + max_batch, num_samples), 3] = (
            output['model_out']
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
#         normals[head : min(head + max_batch, num_samples),:] = diff_operators.gradient(output['model_out'], output['model_in']).detach().cpu()
        head += max_batch
        
        del output['model_out']
        del output['model_in']
        torch.cuda.empty_cache()
        
#     eps = 1e-5
#     idx = np.where(abs(samples[:,3])<eps)[0]
#     pointcloud_stale = np.concatenate([samples[idx,:3],normals[idx,:]], axis=1)
    
        
#     pointcloud_stale[:,3:6] = pointcloud_stale[:,3:6]/np.linalg.norm(pointcloud_stale[:,3:6], axis=1, keepdims = True)
#     write_files(ply_filename, pointcloud_stale)
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    
    
    
    mesh_points = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
        output_return= output_return,
    )
    
    if output_return:
        return mesh_points


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
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

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset
    if scale is not None:
        mesh_points = mesh_points * 1e3

    # try writing to the ply file

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
    with open(output_file, 'a') as f:
        print(pointcloud.shape[0])
        for iidx in range(pointcloud.shape[0]):
            content = " ".join(str(x) for x in pointcloud[iidx,:]) +'\n'
            f.write(content)