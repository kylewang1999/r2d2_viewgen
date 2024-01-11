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
    print('here')

    o3d_pcd = sample_points_from_sphere(radius=5., num_samples=100)
    points_np = np.asarray(o3d_pcd.points)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    vizualizer = o3d.visualization.Visualizer()
    vizualizer.create_window()
    WIDTH, HEIGHT = 480, 640
    vizualizer.create_window(width=WIDTH, height=HEIGHT)

    standardCameraParametersObj = vizualizer.get_view_control().convert_to_pinhole_camera_parameters()
    intrinsic = standardCameraParametersObj.intrinsic.intrinsic_matrix.copy()
    
    camlines = []
    for camera_xyz in points_np:

        # camline = o3d.geometry.LineSet.create_camera_visualization(
        #     view_width_px=WIDTH//2, 
        #     view_height_px=HEIGHT//2, 
        #     intrinsic=intrinsic,
        #     extrinsic=look_at(camera_xyz, np.array([0,0,0]), np.array([0,0,1]))
        # )
        camline = make_frustum(
            size=0.5,
            extrinsic=look_at(camera_xyz, np.array([0,0,0]), np.array([0,0,1])))
        vizualizer.add_geometry(camline)

    vizualizer.add_geometry(coord_frame)
    vizualizer.add_geometry(o3d_pcd)
    vizualizer.get_render_option().point_size = 10.0  
    vizualizer.run()