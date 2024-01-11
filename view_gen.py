'''
File Created:  2024-Jan-10th Wed 1:11
Author:        Kaiyuan Wang (k5wang@ucsd.edu)
Affiliation:   ARCLab @ UCSD
Description:   Main script for visualizing camera extrinsic views
'''
import open3d as o3d
import numpy as np
import open3d.cpu.pybind.geometry as o3d_geo

from utils import make_frustum, look_at, sample_points_from_sphere


if __name__ == '__main__':

    o3d_pcd = sample_points_from_sphere(radius=2., num_samples=100)
    points_np = np.asarray(o3d_pcd.points)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    viz = o3d.visualization.Visualizer()
    WIDTH, HEIGHT = 480, 640
    viz.create_window(width=WIDTH, height=HEIGHT)

    standardCameraParametersObj = viz.get_view_control().convert_to_pinhole_camera_parameters()
    intrinsic = standardCameraParametersObj.intrinsic.intrinsic_matrix.copy()
    
    camlines = []
    for camera_xyz in points_np:
        camline = make_frustum(size=0.2,
            extrinsic=look_at(camera_xyz, np.array([0,0,0]), np.array([0,0,1])))
        viz.add_geometry(camline)
        camlines.append(camline)

    mesh = o3d.io.read_triangle_mesh("assets/all.obj")

    # light = o3d.visualization.rendering.Light()
    # light.intensity = 1000
    # renderer = o3d.visualization.rendering.Renderer()
    # scene = o3d.visualization.rendering.Open3DScene(renderer)
    # viz.get_scene().add_point_light('light', np.array([1,1,1], np.array([5,5,5], 100)))
    # scene.add_light("light", light, np.array([10, 10, 10])) 
    o3d.visualization.draw_geometries([mesh, coord_frame, o3d_pcd, *camlines])
    # viz.add_geometry(mesh)
    # viz.add_geometry(coord_frame)
    # viz.add_geometry(o3d_pcd)
    # viz.get_render_option().point_size = 10.0  
    # viz.run()