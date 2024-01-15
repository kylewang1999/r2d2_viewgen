'''
File Created:  2024-Jan-11th Thu 11:35
Author:        Kaiyuan Wang (k5wang@ucsd.edu)
Affiliation:   ARCLab @ UCSD
Description:   Utility functions for open3d, yaml, file handling, etc.
'''

import yaml, json
import numpy as np, open3d as o3d
import open3d.cpu.pybind.geometry as o3d_geo

from types import SimpleNamespace
# from kornia.geometry.conversions import convert_angles_to_rotation_matrix

def dict_to_namespace(dict_obj):
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            dict_obj[key] = dict_to_namespace(value)
    return SimpleNamespace(**dict_obj)

def parse_yaml_to_namespace(yaml_file:str):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return dict_to_namespace(data)

def parse_extrinsics_json(json_file:str):
    with open(json_file, 'r') as file:
        data = json.load(file)

    all_extrinsics = []
    
    for _, metadata_single_trajectory in data.items():
        for k, v in metadata_single_trajectory.items():
            if 'extrinsic' in k and isinstance(v, list):
                all_extrinsics.append(np.array(v))
    return np.stack(all_extrinsics, axis=0)

def make_extrinsic_matrix(t_vec:np.array, r_vec:np.array) -> np.array:
    assert t_vec.shape == (3,) == r_vec.shape

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz(r_vec)
    extrinsic[:3, 3] = t_vec
    return extrinsic

def sample_points_from_sphere(radius:float=5., num_samples:int=10) -> o3d_geo.PointCloud:

    phi = np.random.uniform(0, 2*np.pi, num_samples*5)
    theta = np.random.uniform(0, np.pi, num_samples*5)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points = np.stack([x,y,z], axis=1)

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)

    o3d_pcd = o3d_pcd.farthest_point_down_sample(num_samples)

    return o3d_pcd

def o3d_pcd_from_numpy(points:np.array):
    assert points.shape[1] == 3
    assert points.ndim == 2

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    return o3d_pcd

def make_frustum(size:float=1.0, extrinsic:np.array=np.eye(4), opacity:float=0.5) -> o3d_geo.LineSet:
    """ Make a oriented frustum with the given size and extrinsic matrix """

    points = size * np.array([[0,0,0], [1,1,1], [1,-1,1], [-1,-1,1], [-1,1,1], [0,1.5,1]])
    lines = np.array([[0,1], [0,2], [0,3], [0,4], [1,2], [2,3], [3,4], [4,1], [5,4], [5,1]])

    # transform the points
    points = np.hstack([points, np.ones((points.shape[0], 1))]) # (N,3) -> (N,4)
    points = extrinsic @ points.T   # (4,4) @ (4,N) -> (4,N)
    points = points.T[:, :3]        # (N,4) -> (N,3)

    lines = o3d_geo.LineSet(
        points = o3d.utility.Vector3dVector(points),
        lines = o3d.utility.Vector2iVector(lines)
    )
    lines.paint_uniform_color(np.array([1,1,1]) * (1-opacity))
    return lines

def look_at(camera_xyz:np.array, target_xyz:np.array, up:np.array=np.array([0,0,1])) -> np.array:
    ''' Generate SE(3) transformation matrix from camera position and target position '''

    assert camera_xyz.shape == (3,)
    assert target_xyz.shape == (3,)
    assert up.shape == (3,)

    z = target_xyz - camera_xyz
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    se3_mat = np.eye(4)
    se3_mat[:3, 0] = x
    se3_mat[:3, 1] = y
    se3_mat[:3, 2] = z
    se3_mat[:3, 3] = camera_xyz
    # se3_mat[:3, :3] = se3_mat[:3, :3].T
    # se3_mat = np.linalg.inv(se3_mat)
    return se3_mat  # (4,4) SE(3) transformation matrix


if __name__ == '__main__':
    
    json_file = 'assets/r2d2_metadata.json'
    extrinsics = parse_extrinsics_json(json_file)

    output_file = 'assets/r2d2_extrinsics.npy'
    np.save(output_file, extrinsics)
    print(f'extrinsics array of shape {extrinsics.shape} is saved to {output_file}')