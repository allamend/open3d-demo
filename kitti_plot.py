# KITTI pointcloud with correspondence plot after applying transformation

import os
import open3d as o3d
import numpy as np
import pickle
import copy

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


result = np.load('/home/yfenghua/dataset/pointcloud/10_1099_1121.npz')
print(result.files)

pc1 = o3d.geometry.PointCloud()
pc2 = o3d.geometry.PointCloud()
pc3 = o3d.geometry.PointCloud()
pc4 = o3d.geometry.PointCloud()
pc1.points = o3d.utility.Vector3dVector(result['src_points'])

# can change the correspondence threshold
thr = 0.7

pc2.points = o3d.utility.Vector3dVector(result['src_corr_points'][result['corr_scores'] > thr])
pc1.translate((20, 10, 10))
pc2.translate((20, 10, 10))
pc3.points = o3d.utility.Vector3dVector(result['ref_points'])
pc4.points = o3d.utility.Vector3dVector(result['ref_corr_points'][result['corr_scores'] > thr])
shape = result['ref_corr_points'][result['corr_scores'] > thr].shape
print(shape)
line_set = o3d.geometry.LineSet()
line_set = line_set.create_from_point_cloud_correspondences(pc2,pc4,[(i ,i) for i in range(shape[0])])
line_set.paint_uniform_color([0,0,1])
# pc3.points = o3d.utility.Vector3dVector(result['ref_points_c'])
pc1.paint_uniform_color([0,0.5,0])
pc2.paint_uniform_color([0,0.9,0])
pc3.paint_uniform_color([0.5,0,0])
pc4.paint_uniform_color([0.9,0,0])
o3d.visualization.draw_geometries([pc1, pc2, pc3, pc4, line_set])