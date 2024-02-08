'''
File Created:  2024-Jan-10th Wed 1:11
Author:        Kaiyuan Wang (k5wang@ucsd.edu)
Affiliation:   ARCLab @ UCSD
Description:   Main script for visualizing camera extrinsic views
'''
import open3d as o3d, numpy as np
import matplotlib.pyplot as plt
import open3d.cpu.pybind.geometry as o3d_geo
import scipy.stats as stats

from utils import *
from mpl_toolkits.mplot3d import Axes3D
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

def load_scene_glb(fname:str='/Users/KyleWang/repos/r2d2_3d_assets/r2d2_kitchen_scene_01.gltf'):
    mesh = o3d.io.read_triangle_mesh(fname)
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
        camline = make_frustum(size=0.1,
            extrinsic=look_at(camera_xyz, np.array([0,0,0]), np.array([0,0,1])))
        viz.add_geometry(camline)
        camlines.append(camline)

    mesh = load_panda_mesh()
    o3d.visualization.draw_geometries([mesh, coord_frame, o3d_pcd, *camlines])


def rotate_view(vis:o3d.visualization.Visualizer):
    ctrl = vis.get_view_control()
    ctrl.rotate(1.0, 0.0)
    return False


def visualize_pose_stats(subsample_criteria:str='furthest', orientation:str='look_at',num_samples:int=100):
    """Visualize the pose statistics of R2D2 dataset
    Args:
    - subsample_criteria: str, 'random' or 'kmeans'
    - orientation: str, 'look_at' or 'recorded'
    """
    assert subsample_criteria in ['furthest', 'random', 'kmeans']
    assert orientation in ['look_at', 'recorded']

    # NOTE: R2D2 Pose convention is [t_vec, r_vec]
    extrinsics = np.load('assets/r2d2_extrinsics.npy')  # (133836, 6) Original -> (89224, 6) Removed hand camera
    extrinsics = np.unique(extrinsics, axis=0)          # (1417, 6) Removed duplicates

    kde = stats.gaussian_kde(extrinsics[:, :3].T)
    densities = kde(extrinsics[:, :3].T)
    
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
    elif subsample_criteria == 'furthest':
        pcd_down = o3d_pcd_from_numpy(xyzs).farthest_point_down_sample(num_samples)
        pcd_down_np = np.asarray(pcd_down.points)
        selected_ids, _ = pairwise_distances_argmin_min(pcd_down_np, xyzs)
    else:
        raise NotImplementedError

    cmap = plt.get_cmap('magma') 
    xyzs_chosen = xyzs[selected_ids]
    ext_mats_chosen = ext_mats[selected_ids]
    densities_chosen = densities[selected_ids]
    colors = cmap(densities_chosen / densities_chosen.max())[:,:3]
    # # For each xyz, find number of neighbors within a certain radius
    # # Then, color the frustum by the number of neighbors
    # radius = 0.1
    # kdtree = o3d.geometry.KDTreeFlann(o3d_pcd_from_numpy(xyzs))
    # num_neighbors = []
    # for point in xyzs_chosen:
    #     _, idx, _ = kdtree.search_radius_vector_3d(point, radius)
    #     num_neighbors.append(len(idx))
    # num_neighbors = np.array(num_neighbors)
    # 
    # colors = cmap(num_neighbors / num_neighbors.max())[:,:3]

    max_dist = np.linalg.norm(xyzs, axis=1).max()

    camlines = []
    for xyz, ext_mat, color in zip(xyzs_chosen, ext_mats_chosen, colors):
        ext1 = look_at(xyz, np.array([0,0,0]), np.array([0,0,1]))
        ext2 = ext_mat
        opacity = 1 - (np.linalg.norm(xyz) / max_dist)
        camlines.append(make_frustum(size=0.03, 
                                     opacity=opacity,
                                     color=color,
                                     geometry_type='mesh',
                                     extrinsic=ext1 if orientation=='look_at' else ext2))
        
    merged_camlines = merge_linesets(camlines) if isinstance(camlines[0], o3d_geo.LineSet) else merge_meshes(camlines)
    
    all_geometries =  [
        load_panda_mesh(),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0]),
        # o3d_pcd_from_numpy(xyzs),
        # o3d_pcd_from_numpy(extrinsics[:,:3]),
        merged_camlines,
        # *camlines,
        # o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    ]

    # export merged_camlines to a file
    fname = 'assets/merged_camlines_true_orientation.ply' if orientation == 'recorded' else \
        'assets/merged_camlines_lookat.ply'
    if isinstance(merged_camlines, o3d_geo.LineSet):
        o3d.io.write_line_set(fname, merged_camlines)
    else:
        o3d.io.write_triangle_mesh(fname, merged_camlines)
    print(f'Wrote to {fname}')
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Using Recorded Orientation" if orientation == 'recorded' else "Enforcing Look-at Orientation")
    vis.get_render_option().line_width = 10.0
    vis.get_render_option().mesh_show_back_face = True
    # for geo in all_geometries: vis.add_geometry(geo)
    vis.add_geometry(o3d.io.read_triangle_mesh(fname))

    ctrl = vis.get_view_control()
    ctrl.set_up(np.array([[0.,0.,1.]]).astype(np.float64).T)
    ctrl.set_front(np.array([[1.,0.,0.]]).astype(np.float64).T)
    ctrl.set_zoom(0.4)

    while True:
        ctrl.rotate(1.0, 0.0)
        vis.poll_events()
        vis.update_renderer()

def test_glb_load():

    fname:str='/Users/KyleWang/repos/r2d2_3d_assets/r2d2_kitchen_scene_01.glb'

    mesh = o3d.io.read_triangle_mesh(fname)
    
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GLB Test")
    vis.add_geometry(pcd)

    ctrl = vis.get_view_control()
    ctrl.set_up(np.array([[0.,0.,1.]]).astype(np.float64).T)
    ctrl.set_front(np.array([[1.,0.,0.]]).astype(np.float64).T)
    ctrl.set_zoom(0.4)

    vis.run()

def test_ply_load(fname):

    mesh = o3d.io.read_triangle_mesh(fname)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Test")
    vis.add_geometry(mesh)

    ctrl = vis.get_view_control()
    ctrl.set_up(np.array([[0.,0.,1.]]).astype(np.float64).T)
    ctrl.set_front(np.array([[1.,0.,0.]]).astype(np.float64).T)
    ctrl.set_zoom(0.4)

    vis.run()

if __name__ == '__main__':

    visualize_pose_stats(subsample_criteria='furthest', orientation='look_at', num_samples=100)

    # test_ply_load('assets/merged_camlines_lookat.ply')