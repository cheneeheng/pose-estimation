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


VERTICAL = False


# BASE_PATH = '/data/realsense_230511/001622070408/230425163246'
# ca = '{BASE_PATH}/calib/dev001622070408_calib.json'
# cf = '{BASE_PATH}/color/1682433172991772498.npy'
# df = '{BASE_PATH}/depth/1682433172991772498.npy'
# CROP_H = [200, -50]
# CROP_W = [380, -420]
# SUBSAMPLE_RAW_DATA = 10
# SUBSAMPLE_ROT_DATA = 30
# Z_VAL_FILTER = -2000
# Y_OFFSET = -3000
# print(plane)
# # [-0.74005933  0.          0.03845016] 0.8346439656990158
# print(axis, angle)
# # axis = [-0.740, 0., -0.01]
# # angle = 0.83
# print(matrix_z)
# # [[ 0.99911548 -0.03845016 -0.01702447]
# #  [ 0.03845016  0.67144157  0.74005933]
# #  [-0.01702447 -0.74005933  0.67232608]]

# BASE_PATH = '/data/realsense_230511/001622070408/230511160519'
# ca = f'{BASE_PATH}/calib/dev001622070408_calib.json'
# cf = f'{BASE_PATH}/color/1683813923642666251.npy'
# df = f'{BASE_PATH}/depth/1683813923642666251.npy'
# CROP_H = [50, -10]
# CROP_W = [500, -300]
# SUBSAMPLE_RAW_DATA = 10
# SUBSAMPLE_ROT_DATA = 30
# Z_VAL_FILTER = 3500
# Y_OFFSET = -1500
# Z_OFFSET = 1000
# # Plane(point=Point([ 699.89866096, 1109.52260625, 2891.24285714]),
# #       normal=Vector([0.03090854, 0.92122211, 0.38780727]))
# # [-0.38780727  0.          0.03090854] 0.3995860521451204
# # [[ 0.99950274 -0.03090854 -0.00623903]
# #  [ 0.03090854  0.92122211  0.38780727]
# #  [-0.00623903 -0.38780727  0.92171937]]

# BASE_PATH = '/data/realsense_230516/001622070408/230516115209'
# ca = f'{BASE_PATH}/calib/dev001622070408_calib.json'
# cf = f'{BASE_PATH}/color/1684230752669269699.npy'
# df = f'{BASE_PATH}/depth/1684230752669269699.npy'
# CROP_H = [250, -1]
# CROP_W = [380, -330]
# SUBSAMPLE_RAW_DATA = 3
# SUBSAMPLE_ROT_DATA = 30
# Z_VAL_FILTER = 3500
# X_OFFSET = 0
# Y_OFFSET = -2500
# Z_OFFSET = 1000
# # Plane(point=Point([ 185.8072827 , 1077.01253391, 2822.82516531]),
# #       normal=Vector([0.02967045, 0.9102765 , 0.41293627]))
# # [-0.41293627  0.          0.02967045] 0.42684489335927084
# # [[ 0.99953916 -0.02967045 -0.00641373]
# #  [ 0.02967045  0.9102765   0.41293627]
# #  [-0.00641373 -0.41293627  0.91073734]]

# BASE_PATH = '/data/realsense_230517/001622070408/230517152113/'
# ca = f'{BASE_PATH}/calib/dev001622070408_calib.json'
# cf = f'{BASE_PATH}/color/1684329686682402805.npy'
# df = f'{BASE_PATH}/depth/1684329686682402805.npy'
# CROP_H = [300, -1]
# CROP_W = [380, -330]
# SUBSAMPLE_RAW_DATA = 3
# SUBSAMPLE_ROT_DATA = 30
# Z_VAL_FILTER = 3500
# X_OFFSET = -700
# Y_OFFSET = -1700
# Z_OFFSET = 800
# # Plane(point=Point([ 186.21031656, 1036.51478805, 2828.55338305]),
# #       normal=Vector([0.08064464, 0.90048104, 0.42735271]))
# # [-0.42735271  0.          0.08064464] 0.4499219701364414
# # [[ 0.99657794 -0.08064464 -0.0181342 ]
# #  [ 0.08064464  0.90048104  0.42735271]
# #  [-0.0181342  -0.42735271  0.9039031 ]]

# BASE_PATH = '/data/realsense_230522/001622070408/230522164119/'
# ca = f'{BASE_PATH}/calib/dev001622070408_calib.json'
# cf = f'{BASE_PATH}/color/1684766493048129434.npy'
# df = f'{BASE_PATH}/depth/1684766493048129434.npy'
# CROP_H = [130, -150]
# CROP_W = [550, -1]
# SUBSAMPLE_RAW_DATA = 10
# SUBSAMPLE_ROT_DATA = 10
# Z_VAL_FILTER = 3500
# X_OFFSET = -1600
# Y_OFFSET = -1000
# Z_OFFSET = 1500
# VERTICAL = False
# Plane(point=Point([1382.65052154,  -46.48189615, 2258.12671717]),
#       normal=Vector([ 0.94567705, -0.01504597,  0.32475919]))
# [-0.32475919  0.          0.94567705] 1.585842865928153
# [[ 0.09203368 -0.94567705 -0.31180878]
#  [ 0.94567705 -0.01504597  0.32475919]
#  [-0.31180878 -0.32475919  0.89292035]]

# BASE_PATH = '/data/realsense_230523/001622070408/230523101302/'
# ca = f'{BASE_PATH}/calib/dev001622070408_calib.json'
# cf = f'{BASE_PATH}/color/1684829585599577116.npy'
# df = f'{BASE_PATH}/depth/1684829585599577116.npy'
# CROP_H = [200, -180]
# CROP_W = [550, -1]
# SUBSAMPLE_RAW_DATA = 10
# SUBSAMPLE_ROT_DATA = 10
# Z_VAL_FILTER = 3500
# # X_OFFSET = -2200
# # Y_OFFSET = -700
# # Z_OFFSET = 0
# X_OFFSET = -2200
# Y_OFFSET = -700
# Z_OFFSET = 0
# VERTICAL = False
# # Plane(point=Point([1386.13870592,   60.35210064, 2254.42350168]),
# #       normal=Vector([ 0.9443577 , -0.0202894 ,  0.32829389]))
# # [-0.32829389  0.          0.9443577 ] 1.591087123479195
# # [[ 0.08971949 -0.9443577  -0.31644739]
# #  [ 0.9443577  -0.0202894   0.32829389]
# #  [-0.31644739 -0.32829389  0.88999111]]

BASE_PATH = '/data/realsense_230531/001622070408/230531151854/'
ca = f'{BASE_PATH}/calib/dev001622070408_calib.json'
cf = f'{BASE_PATH}/color/1685539138629011904.npy'
df = f'{BASE_PATH}/depth/1685539138629011904.npy'
CROP_H = [200, -180]
CROP_W = [550, -1]
SUBSAMPLE_RAW_DATA = 1
SUBSAMPLE_ROT_DATA = 10
Z_VAL_FILTER = 3500
# X_OFFSET = -2200
# Y_OFFSET = -700
# Z_OFFSET = 0
X_OFFSET = -2200
Y_OFFSET = -700
Z_OFFSET = 0
VERTICAL = False
# Plane(point=Point([1394.538405  ,   60.69744184, 2268.97181818]),
#       normal=Vector([ 0.94602187, -0.01580074,  0.32371742]))
# [-0.32371742  0.          0.94602187] 1.5865977250406023
# [[ 0.09067461 -0.94602187 -0.31116032]
#  [ 0.94602187 -0.01580074  0.32371742]
#  [-0.31116032 -0.32371742  0.89352464]]

# Read data and convert depth --------------------------------------------------
with open(ca) as f:
    calib = json.load(f)
h_d = calib['depth']['height']
w_d = calib['depth']['width']
intr_mat = calib['depth']['intrinsic_mat']
depth_scale = calib['depth']['depth_scale']
color_image = read_color_file(cf, calib['color']['format'])
color_image = color_image.reshape((h_d, w_d, -1))
depth_image = read_depth_file(df)
try:
    depth_image = depth_image[-h_d*w_d:].reshape(h_d, w_d)
except ValueError:
    depth_image = depth_image[-h_d*w_d//4:].reshape(h_d//2, w_d//2)
depth_image = cv2.resize(depth_image, (color_image.shape[1],
                                       color_image.shape[0]))
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
if VERTICAL:
    x3d_mm = np.rot90(x3d_mm, -1)
    y3d_mm = np.rot90(y3d_mm, -1)
    z3d_mm = np.rot90(z3d_mm, -1)
    color_image = np.rot90(color_image, -1)
    depth_image = np.rot90(depth_image, -1)

if False:
    # depth_image = np.clip(depth_image, 0/depth_scale, 5/depth_scale)*2
    # depth_colormap = cv2.applyColorMap(
    #     cv2.convertScaleAbs(depth_image, alpha=0.03),
    #     cv2.COLORMAP_JET)
    # cv2.imshow("000", depth_colormap)
    cv2.imshow("001", color_image)
    cv2.waitKey(0)
    exit(1)


# Prepare plotting figure ------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.view_init(elev=-30, azim=10, roll=90)
# ax.view_init(-90, -90, 0)
ax.view_init(-90, 180, 90)
ax.set_xlim(-5000, 5000)
ax.set_ylim(-5000, 5000)
ax.set_zlim(-3000, 5000)
ax.set_xlabel("x")
ax.set_ylabel("y")

# PPlane fitting ---------------------------------------------------------------
plane_x3d_mm = x3d_mm[CROP_H[0]:CROP_H[1], CROP_W[0]:CROP_W[1]]
plane_y3d_mm = y3d_mm[CROP_H[0]:CROP_H[1], CROP_W[0]:CROP_W[1]]
plane_z3d_mm = z3d_mm[CROP_H[0]:CROP_H[1], CROP_W[0]:CROP_W[1]]
plane_color_image = color_image[CROP_H[0]:CROP_H[1], CROP_W[0]:CROP_W[1]]
plane_data = np.stack([plane_x3d_mm, plane_y3d_mm, plane_z3d_mm],
                      axis=-1).reshape(-1, 3)[::SUBSAMPLE_RAW_DATA]
plane_color_data = np.flip(plane_color_image.reshape(-1, 3), axis=-1
                           )[::SUBSAMPLE_RAW_DATA]
plane_color_data = plane_color_data[plane_data[:, 2] < Z_VAL_FILTER]
plane_data = plane_data[plane_data[:, 2] < Z_VAL_FILTER]
plane_color_data = plane_color_data[plane_data[:, 2] > 0.1]
plane_data = plane_data[plane_data[:, 2] > 0.1]
print(f"#data : {len(plane_data)}")

points = Points(plane_data)
plane = Plane.best_fit(points)

if False:
    _, ax = plot_3d(
        points.plotter(c=plane_color_data/255, s=3, depthshade=False),
        plane.plotter(alpha=0.5, lims_x=(-200, 200), lims_y=(-200, 200)),
    )
    plt.show()
    exit(1)

axis = np.cross(plane.normal, [0, 1, 0])
angle = angle_between(plane.normal, [0, 1, 0])
matrix_z = rotation_matrix(axis, angle)
print(plane)
print(axis, angle)
print(matrix_z)

# sheer_h = np.eye(3)
# sheer_h[0, 2] = 0.0
# sheer_h[1, 2] = 0.0
# print(sheer_h)

data_rot = np.stack(
    [x3d_mm, y3d_mm, z3d_mm], axis=-1).reshape(-1, 3)[::SUBSAMPLE_ROT_DATA]
data_rot += np.array([X_OFFSET, Y_OFFSET, Z_OFFSET])
data_rot = (matrix_z@data_rot.transpose()).transpose()
# data_rot[:, 0] *= -1
# data_rot[:, 1] *= -1
color_data = np.flip(color_image.reshape(-1, 3), axis=-1)[::SUBSAMPLE_ROT_DATA]
color_data = color_data[data_rot[:, 2] < 5000]
data_rot = data_rot[data_rot[:, 2] < 5000]
ax.scatter(-data_rot[:, 0], data_rot[:, 1],
           data_rot[:, 2], c=color_data/255, marker='.')
plt.tight_layout()
plt.show()
