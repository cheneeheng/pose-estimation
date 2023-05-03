from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import json
from rs_py.utility import read_color_file
from rs_py.utility import read_depth_file
import cv2
from tqdm import trange

from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d

from data_gen.rotation import angle_between
from data_gen.rotation import rotation_matrix

ca = '/data/realsense/001622070408/230425163246/calib/dev001622070408_calib.json'  # noqa
cf = '/data/realsense/001622070408/230425163246/color/1682433172991772498.npy'
df = '/data/realsense/001622070408/230425163246/depth/1682433172991772498.npy'
# cf = '/data/realsense/001622070408/230425163246/color/1682433169525577915.npy'
# df = '/data/realsense/001622070408/230425163246/depth/1682433169525577915.npy'
with open(ca) as f:
    calib = json.load(f)
h_d = calib['depth']['height']
w_d = calib['depth']['width']
intr_mat = calib['depth']['intrinsic_mat']
depth_scale = calib['depth']['depth_scale']
color_image = read_color_file(cf, calib['color']['format'])
color_image = color_image.reshape((h_d, w_d, -1))
depth_image = read_depth_file(df)
depth_image = depth_image[-h_d*w_d:].reshape(h_d, w_d)
depth_image = np.fromfile('/data/tmp/depth_image_filter.bin', dtype=np.uint16)
depth_image = depth_image.reshape(h_d, w_d)
h_d, w_d = depth_image.shape
x, y = np.meshgrid(np.linspace(0, w_d, w_d), np.linspace(0, h_d, h_d))
fx = intr_mat[0]
fy = intr_mat[4]
cx = intr_mat[2]
cy = intr_mat[5]
x3d_mm = (x-cx) / fx * depth_image
y3d_mm = (y-cy) / fy * depth_image
z3d_mm = depth_image.astype(float)
# print(np.median(z3d_mm))
# z3d_mm -= np.median(z3d_mm)

depth_image = np.clip(depth_image, 0/depth_scale, 5/depth_scale)*2
depth_colormap = cv2.applyColorMap(
    cv2.convertScaleAbs(depth_image, alpha=0.03),
    cv2.COLORMAP_JET)
cv2.imshow("000", depth_colormap)
cv2.waitKey(0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.view_init(elev=-30, azim=10, roll=90)
ax.view_init(-90, -90, 0)
ax.view_init(-90, 180, 90)
ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_zlim(3000, 5000)
ax.set_xlabel("x")
ax.set_ylabel("y")

plane_x3d_mm = x3d_mm[200:-50, 380:-420]
plane_y3d_mm = y3d_mm[200:-50, 380:-420]
plane_z3d_mm = z3d_mm[200:-50, 380:-420]
plane_color_image = color_image[200:-50, 380:-420]
plane_data = np.stack([plane_x3d_mm, plane_y3d_mm, plane_z3d_mm], axis=-1).reshape(-1, 3)[::10]  # noqa
plane_color_data = np.flip(plane_color_image.reshape(-1, 3), axis=-1)[::10]
plane_color_data = plane_color_data[plane_data[:, 2] > -2000]
plane_data = plane_data[plane_data[:, 2] > -2000]
print(f"#data        : {len(plane_data)}")

points = Points(plane_data)
plane = Plane.best_fit(points)
# plot_3d(
#     points.plotter(c='r', s=3, depthshade=False),
#     plane.plotter(alpha=0.5, lims_x=(-5000, 5000), lims_y=(-5000, 5000)),
# )
# plt.show()
print(plane)
# [-0.74005933  0.          0.03845016] 0.8346439656990158

plane.normal
axis = np.cross(plane.normal, [0, 1, 0])
angle = angle_between(plane.normal, [0, 1, 0])
print(axis, angle)
# axis = [-0.740, 0., -0.01]
# angle = 0.83

# axis = np.cross(eq[:-1], [0, -1, 0])
# angle = angle_between(eq[:-1], [0, -1, 0])
# print(axis, angle)

# axis = [-1, 0, 0]
# angle = 0.8

# [[ 0.99911548 -0.03845016 -0.01702447]
#  [ 0.03845016  0.67144157  0.74005933]
#  [-0.01702447 -0.74005933  0.67232608]]
matrix_z = rotation_matrix(axis, angle)
print(matrix_z)
data_rot = np.stack([x3d_mm, y3d_mm, z3d_mm], axis=-1).reshape(-1, 3)[::50]  # noqa
data_rot += np.array([0, -3000, 500])
data_rot = (matrix_z@data_rot.transpose()).transpose()
# data_rot[:, 0] *= -1
data_rot[:, 1] *= 1.5
color_data = np.flip(color_image.reshape(-1, 3), axis=-1)[::50]
ax.scatter(data_rot[:, 0], data_rot[:, 1],
           data_rot[:, 2], c=color_data/255, marker='.')
plt.tight_layout()
plt.show()
