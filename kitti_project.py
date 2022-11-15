# python code to project KITTI pointcloud back to corresponding images


import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# read images, pointclouds and corresponding calibrations
img = '/home/yfenghua/dataset/pointcloud/001099.png'
pc = np.load('/home/yfenghua/dataset/pointcloud/10_1099_1121.npz')
points = pc['src_points']
with open(f'/home/yfenghua/dataset/pointcloud/calib.txt','r') as f:
    calib = f.readlines()

# P2 (3 x 4) for left eye
P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
# Add a 1 in bottom-right, reshape to 4 x 4
R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

velo = np.insert(points,3,1,axis=1).T
velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
cam = P2 * R0_rect * Tr_velo_to_cam * velo
cam = np.delete(cam,np.where(cam[2,:]<0)[1],axis=1)
# get u,v,z
cam[:2] /= cam[2,:]
# do projection staff
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
plt.imshow(png)
# filter point out of canvas
u,v,z = cam
u_out = np.logical_or(u<0, u>IMG_W)
v_out = np.logical_or(v<0, v>IMG_H)
outlier = np.logical_or(u_out, v_out)
cam = np.delete(cam,np.where(outlier),axis=1)
# generate color map from depth, can also change color visualization based on other features
u,v,z = cam
plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
plt.title('result')
# plt.savefig(f'./data_object_image_2/testing/projection/{name}.png',bbox_inches='tight')
plt.show()