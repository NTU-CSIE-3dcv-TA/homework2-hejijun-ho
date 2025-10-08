import open3d as o3d
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os

def load_point_cloud(points3D_df):
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB']) / 255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0],
                                              [1, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 1]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 1]])
    return axes

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat

def apply_transform(points, rotation, translation, scale):
    transform_mat = get_transform_mat(rotation, translation, scale)
    transformed = (transform_mat @ np.concatenate([
                        points.T,
                        np.ones([1, points.shape[0]])
                    ], axis=0)).T
    return transformed

# ---------------------- 可直接修改的控制變數 ----------------------
R_euler = np.array([0, 0, 0])       # rotation (degrees)   // bak
t = np.array([0.5, -0.3, 0.5])      # translation (x, y, z)
scale = 0.5                         # scale

# R_euler = np.array([0, 0, 0])       # rotation (degrees)
# t = np.array([1, -0.3, -1.2])      # translation (x, y, z)
# scale = 0.4                         # scale
# ----------------------------------------------------------------

# 載入資料
points3D_df = pd.read_pickle("data/points3D.pkl")
pcd = load_point_cloud(points3D_df)
axes = load_axes()

# === 建立 cube (原本大小) ===
cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
cube_vertices = np.asarray(cube.vertices).copy()

# === 建立 7×7×7 網格點 (在 [0,1]^3 範圍內) ===
edge = 10
grid_range = np.linspace(0, 1, edge)
gx, gy, gz = np.meshgrid(grid_range, grid_range, grid_range)
grid_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)  # (343, 3)

# === 只保留 cube 外部點 (邊界點) ===
mask = (
    (np.isclose(grid_points[:, 0], 0) | np.isclose(grid_points[:, 0], 1)) |
    (np.isclose(grid_points[:, 1], 0) | np.isclose(grid_points[:, 1], 1)) |
    (np.isclose(grid_points[:, 2], 0) | np.isclose(grid_points[:, 2], 1))
)
outer_points = grid_points[mask]  # 約去除內部 (保留外殼點)

# === 每個面的顏色 (RGB) ===
face_colors = {
    'x0': [1, 0, 0],   # red
    'x1': [0, 1, 0],   # green
    'y0': [0, 0, 1],   # blue
    'y1': [1, 1, 0],   # yellow
    'z0': [1, 0, 1],   # magenta
    'z1': [0, 1, 1]    # cyan
}

# === 為每個點分配顏色 ===
colors = np.zeros_like(outer_points)
for i, p in enumerate(outer_points):
    if np.isclose(p[0], 0):
        colors[i] = face_colors['x0']
    elif np.isclose(p[0], 1):
        colors[i] = face_colors['x1']
    elif np.isclose(p[1], 0):
        colors[i] = face_colors['y0']
    elif np.isclose(p[1], 1):
        colors[i] = face_colors['y1']
    elif np.isclose(p[2], 0):
        colors[i] = face_colors['z0']
    elif np.isclose(p[2], 1):
        colors[i] = face_colors['z1']

# === 應用旋轉、平移、縮放 ===
transformed_points = apply_transform(outer_points, R_euler, t, scale)

# === 根據 z 值由大到小排序 ===
sort_idx = np.argsort(-transformed_points[:, 2])
transformed_points_sorted = transformed_points[sort_idx]
colors_sorted = colors[sort_idx]

# === 顯示 cube + point cloud + grid points ===
cube.vertices = o3d.utility.Vector3dVector(apply_transform(cube_vertices, R_euler, t, scale))
cube.compute_vertex_normals()
cube.paint_uniform_color([1, 0.706, 0])

points_obj = o3d.geometry.PointCloud()
points_obj.points = o3d.utility.Vector3dVector(transformed_points_sorted)
points_obj.colors = o3d.utility.Vector3dVector(colors_sorted)

o3d.visualization.draw_geometries([pcd, axes, cube, points_obj])

# === 儲存結果 ===
save_path_pts = os.path.join(os.getcwd(), "cube_transformed_vertices.npy")
save_path_color = os.path.join(os.getcwd(), "cube_color.npy")
np.save(save_path_pts, transformed_points_sorted)
np.save(save_path_color, colors_sorted)

print(f"[✅] Cube surface points saved to:\n{save_path_pts}")
print(f"[✅] Cube colors saved to:\n{save_path_color}")
print(f"[ℹ️] Shape = {transformed_points_sorted.shape}, Colors = {colors_sorted.shape}")

print(np.load("cube_color.npy"))