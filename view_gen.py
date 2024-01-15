'''
File Created:  2024-Jan-10th Wed 1:11
Author:        Kaiyuan Wang (k5wang@ucsd.edu)
Affiliation:   ARCLab @ UCSD
Description:   Main script for visualizing camera extrinsic views
'''
from curses.ascii import ctrl
import open3d as o3d, numpy as np, torch
from sympy import closest_points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d.cpu.pybind.geometry as o3d_geo

from utils import make_frustum, look_at, make_extrinsic_matrix, sample_points_from_sphere, o3d_pcd_from_numpy
from kornia.geometry.conversions import quaternion_from_euler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min

def load_panda_mesh(mesh_file:str='assets/panda.obj') -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    # Switch y and z axis so z points to the up direction, not the depth direction
    mesh = mesh.transform(np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]))
    return mesh

def run_toy_example():
    o3d_pcd = sample_points_from_sphere(radius=2., num_samples=100)
    points_np = np.asarray(o3d_pcd.points)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    viz = o3d.visualization.Visualizer()
    WIDTH, HEIGHT = 480, 640
    viz.create_window(width=WIDTH, height=HEIGHT)
    
    camlines = []
    for camera_xyz in points_np:
        camline = make_frustum(size=0.2,
            extrinsic=look_at(camera_xyz, np.array([0,0,0]), np.array([0,0,1])))
        viz.add_geometry(camline)
        camlines.append(camline)

    mesh = load_panda_mesh()
    o3d.visualization.draw_geometries([mesh, coord_frame, o3d_pcd, *camlines])


def rotate_view(vis:o3d.visualization.Visualizer):
    ctrl = vis.get_view_control()
    ctrl.rotate(1.0, 0.0)
    return False


def visualize_pose_stats(subsample_criteria: str='random', num_samples: int=100):
    """Visualize the pose statistics of R2D2 dataset
    Args:
    - subsample_criteria: str, 'random' or 'kmeans'
    """
    assert subsample_criteria in ['random', 'kmeans']

    # NOTE: R2D2 Pose convention is [t_vec, r_vec]
    extrinsics = np.load('assets/r2d2_extrinsics.npy')  # (133836, 6)
    ext_mats, camlines = [], []

    xyzs = extrinsics[:,:3]
    row_pitch_yaws = extrinsics[:,3:]

    for t_vec, r_vec  in zip(xyzs, row_pitch_yaws):
        ext_mat = make_extrinsic_matrix(t_vec, r_vec)   # (4,4)
        ext_mats.append(ext_mat)
    ext_mats = np.stack(ext_mats, axis=0)

    if subsample_criteria == 'kmeans':
        kmeans = KMeans(n_clusters=num_samples)
        kmeans.fit(xyzs)
        centroids = kmeans.cluster_centers_
        selected_ids, _ = pairwise_distances_argmin_min(centroids, xyzs)
    elif subsample_criteria == 'random':
        selected_ids = np.random.choice(len(xyzs), num_samples, replace=False)
    else:
        raise NotImplementedError

    xyzs = xyzs[selected_ids]
    max_dist = np.linalg.norm(xyzs, axis=1).max()

    camlines = []
    for xyz, ext_mat in zip(xyzs, ext_mats[selected_ids]):
        ext1 = look_at(xyz, np.array([0,0,0]), np.array([0,0,1]))
        ext2 = ext_mat
        opacity = 1 - (np.linalg.norm(xyz) / max_dist)
        camlines.append(make_frustum(size=0.05, extrinsic=ext1, opacity=opacity))
        
    all_geometries =  [
        load_panda_mesh(), 
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0]),
        o3d_pcd_from_numpy(xyzs),
        *camlines
    ]
    
    vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="Using Recorded Orientation")
    vis.create_window(window_name="Enforcing Look-at Orientation")
    vis.get_render_option().line_width = 10.0
    for geo in all_geometries: vis.add_geometry(geo)

    ctrl = vis.get_view_control()
    ctrl.set_up(np.array([[0.,0.,1.]]).astype(np.float64).T)
    ctrl.set_front(np.array([[1.,0.,0.]]).astype(np.float64).T)
    ctrl.set_zoom(0.4)

    while True:
        ctrl.rotate(3.0, 0.0)
        vis.poll_events()
        vis.update_renderer()


if __name__ == '__main__':

    # run_toy_example()
    visualize_pose_stats(subsample_criteria='random', num_samples=50)
    
