# demo code for open3d pointcloud visualization

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

thr = 0.7
pc_st = result['src_corr_points'][result['corr_scores'] > thr]
pc_ed = result['ref_corr_points'][result['corr_scores'] > thr]

st = o3d.geometry.PointCloud()
st_full = o3d.geometry.PointCloud()
st.points = o3d.utility.Vector3dVector(result['src_points_c'])
st_full.points = o3d.utility.Vector3dVector(pc_st)
ed = o3d.geometry.PointCloud()
ed_full = o3d.geometry.PointCloud()
ed.points = o3d.utility.Vector3dVector(result['ref_points_c'])
ed_full.points = o3d.utility.Vector3dVector(pc_ed)
st.paint_uniform_color([1,0,0])
st_full.paint_uniform_color([0.5, 0, 0])
ed.paint_uniform_color([0,1,0])
ed_full.paint_uniform_color([0,0.5,0])
st_trans = copy.deepcopy(st_full).transform(result['transform'])
st_trans.paint_uniform_color([0.7, 0, 0])
o3d.visualization.draw_geometries([st_full, ed_full, st_trans])