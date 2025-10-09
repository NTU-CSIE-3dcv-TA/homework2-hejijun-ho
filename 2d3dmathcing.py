from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
from tqdm import tqdm

import open3d as o3d
import os

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)  # same point ID regarded as a group
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

# def pnpsolver(query,model,cameraMatrix=0,distortion=0):   ## retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))   # (2D, 3D)
#     kp_query, desc_query = query
#     kp_model, desc_model = model
#     cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
#     distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

#     # TODO: solve PnP problem using OpenCV
#     # Hint: you may use "Descriptors Matching and ratio test" first
#     return None, None, None, None

def pnpsolver(query, model, cameraMatrix=0, distortion=0):  #query: ([XY1, XY2, ...], [des1, des2, des3, ...])  model: ([XYZ1, XYZ2, ...], [des1, des2, des3, ...])
    """
    Solve Perspective-n-Point (PnP) using feature matching and RANSAC.
    query: (kp_query, desc_query) from the query image  (2D keypoints & descriptors)
    model: (kp_model, desc_model) from the 3D model (3D points & descriptors)
    """
    # kp_query, desc_query, corres_3d_query = query
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27, 0, 540],
                             [0, 1869.18, 960],
                             [0, 0, 1]], dtype=np.float32)
    distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352], dtype=np.float32)

    # --- Step 1: 建立匹配器並進行 knn 比對 (k=2)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc_query, desc_model, k=2)
    # print(f"matches: {type(matches)}")

    # --- Step 2: Ratio Test (Lowe’s ratio test)
    good_matches = []
    for m, n in matches:  # 2nd good is for removing faked match
        if m.distance < 0.3 * n.distance:  # 0.75 0.6 0.3
            good_matches.append(m)
    # print("matches: ", good_matches)

    if len(good_matches) < 6:
        # 太少匹配無法求解
        print("NO enough match pairs !!!")
        return None, None, None, None

    # --- Step 3: 根據匹配取出 2D–3D 對應
    img_points = []
    for m in good_matches:
        img_points.append(kp_query[m.queryIdx])
    img_points = np.array(img_points, dtype=np.float32)
    
    obj_points = []
    for m in good_matches:
        obj_points.append(kp_model[m.trainIdx])
    obj_points = np.array(obj_points, dtype=np.float32)

    # --- Step 4: 使用 RANSAC 的 solvePnP 求解姿態
    self_done = False
    if self_done:
        self_done = True
        ################# implement 4 bonus here
    else:
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(  # with ransac
            objectPoints=obj_points,
            imagePoints=img_points,
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs,
            iterationsCount=1000,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    if not retval:
        return None, None, None, None

    # print(f"~~~~\nretval: {retval}\nrvec: {rvec}\ntvec: {tvec}\ninliers: {inliers}\n~~~~")
    return retval, rvec, tvec, inliers              ## R, _ = cv2.Rodrigues(rvec) 轉成旋轉矩陣


def mult_quaternion(q1, q2): # q: list [x, y, z, w]
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return [x, y, z, w]

def len_quaternion(q):
    len = 0
    for i in range(4):
        len += q[i]*q[i]
    return len ** 0.5

def rotation_error(q1, q2): # q2 is ground truth / q = [x, y, z, w] (w + xi + yj + zk)
    #TODO: calculate rotation error # q2 q1*
    len_q1 = len_quaternion(q1)
    len_q2 = len_quaternion(q2)
    # print(f"len: q1<{len_q1}>, q2<{len_q2}>")
    for i in range(4):
        q1[i] = q1[i] / len_q1
        q2[i] = q2[i] / len_q2
    # q1 = q1 / len_q1
    # q2 = q2 / len_q2  ## normalized
    
    q1 = -1 * np.array(q1)
    q1[3] = -q1[3]
    q = mult_quaternion(list(q1), q2)
    return (q[0], q[1], q[2], 2 * np.arccos(np.clip(q[3], -1.0, 1.0)))   # (axis, radians)

def translation_error(t1, t2):
    #TODO: calculate translation error
    dim = 3
    error = 0
    for i in range(dim):
        error += (t1[i] - t2[i]) ** 2
    error = error ** 0.5
    return error

def quadrangular(R, T, scale=0.1):
    """
    Compute the 3D coordinates of a camera's quadrangular pyramid (frustum base).
    Args:
        R: (3x3) rotation matrix from solvePnP
        T: (3x1) translation vector from solvePnP
        scale: size scaling factor (same unit as T)
    Returns:
        apex: (3,) camera optical center in world coordinates
        corners: (4,3) four base corners in world coordinates
    """
    R = np.asarray(R).reshape(3, 3)
    T = np.asarray(T).reshape(3, 1)
    # 相機中心 C = -R^T * T
    apex = (-R.T @ T).flatten()
    # 四角錐在相機座標系下的底面角點
    half = scale * 0.5
    pts_cam = np.array([
        [ half,  half, scale],
        [ half, -half, scale],
        [-half, -half, scale],
        [-half,  half, scale],
    ])
    # 轉到世界座標
    corners = (R.T @ pts_cam.T).T + apex.reshape(1, 3)
    return apex, corners


def visualization(Camera2World_Transform_Matrixs, points3D_df, scale=0.5):
    #TODO: visualize the camera pose
    # pass
    """
    Visualize 3D points and camera pyramids using Open3D.
    Arguments:
        Camera2World_Transform_Matrixs: list of (4x4) np.ndarray
        points3D_df: pandas DataFrame with 3D points (columns: XYZ or X,Y,Z)
        scale: size of each camera pyramid
    """
    geometries = []
    # --- 1. 3D points (gray point cloud) ---
    pts = np.array(points3D_df['XYZ'].to_list())
    # print("#####", points3D_df['RGB'].to_list(), "!!!!!")
    rgb = np.array(points3D_df["RGB"].to_list()) / 255.0
    if pts.size > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # gray = np.ones_like(pts) * 0.6              # np.array([[0.6,0.6,0.6], [0.6,0.6,0.6], ...])  pts.size * 3
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        geometries.append(pcd)

    # --- 2. Camera pyramids ---
    trajectory_apex = []
    trajectory = []
    trajectory_colors = []
    for i, mat in enumerate(Camera2World_Transform_Matrixs):
        mat = np.asarray(mat)
        # Extract rotation and translation (camera pose)
        R_c2w = mat[:3, :3]
        C = mat[:3, 3].reshape(3, 1)
        R = R_c2w.T
        T = -R @ C
        
        apex, corners = quadrangular(R, T, scale=scale)
        vertices = np.vstack([apex, corners])  # apex=0, corners=4
        bottom_triangles = np.array([
            [1, 2, 3],
            [1, 3, 4],
            [3, 2, 1],
            [4, 3, 1]
        ])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(bottom_triangles)
        mesh.paint_uniform_color([0.6, 0.8, 1.0])
        mesh.compute_vertex_normals()
        geometries.append(mesh)
        lines = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ])
        line_colors = [[0.2, 0.6, 1.0] for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        geometries.append(line_set)

        # Add camera center (red sphere)
        cam_center = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.05)
        cam_center.paint_uniform_color([1, 0, 0])
        cam_center.translate(apex)
        geometries.append(cam_center)

        # Add direction arrow (camera forward axis)
        z_axis = o3d.geometry.TriangleMesh.create_arrow(
            cone_radius=scale * 0.05,
            cone_height=scale * 0.2,
            cylinder_radius=scale * 0.02,
            cylinder_height=scale * 1.1 #0.3
        )
        z_axis.paint_uniform_color([1, 0, 0])
        z_axis.rotate(R_c2w, center=(0, 0, 0))
        z_axis.translate(C.flatten())
        geometries.append(z_axis)
        # add trajectory here
        trajectory_apex.append(apex)
        aplength = len(trajectory_apex)
        if aplength != 1:
            trajectory.append([aplength-1, aplength-2])            
            trajectory_colors.append([1, 0, 0])
    trajectory = np.array(trajectory)
    trajectory_set = o3d.geometry.LineSet()
    trajectory_set.points = o3d.utility.Vector3dVector(trajectory_apex)
    trajectory_set.lines = o3d.utility.Vector2iVector(trajectory)
    trajectory_set.colors = o3d.utility.Vector3dVector(trajectory_colors)
    geometries.append(trajectory_set)

    # o3d.io.write_point_cloud("scene_points.ply", pcd)
    # o3d.io.write_triangle_mesh("scene_mesh.ply", mesh)
    # geometries.append(line_set)  geometries.append(cam_center)   geometries.append(z_axis)  # store to .ply

    # --- visualize all geometries ---
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Camera Pyramids & 3D Points",
        width=1200,
        height=800,
        mesh_show_back_face=True
    )

def world2camera2image(c2w, p_world): # c2w: camera to world matrix(4*4), p_world: world coordonate system
    cameraMatrix = np.array([[1868.27, 0, 540],
                             [0, 1869.18, 960],
                             [0, 0, 1]], dtype=np.float32)
    distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352], dtype=np.float32)
    # c2w = [R_c2w | t_c2w]   =>  w2c = [R_c2w.T | -R_c2w.T @ t_c2w]
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3].reshape(3, 1)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    p_world = np.asarray(p_world, dtype=np.float32).reshape(-1, 3)
    p_cam = (R_w2c @ p_world.T + t_w2c).T  # (N,3)
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c
    p_img, _ = cv2.projectPoints(p_world, rvec, tvec, cameraMatrix, distCoeffs)
    p_img = p_img.reshape(-1, 2)
    return p_img

def sort_depth(c2w, pts_3d, pts_color):  #pts_3d: world coordinate system
    # deep to shallow (Z in camera coordinate system)
    pts_3d = np.asarray(pts_3d, dtype=np.float32).reshape(-1, 3)
    pts_color = np.asarray(pts_color, dtype=np.float32).reshape(-1, 3)
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3].reshape(3, 1)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    pts_cam = (R_w2c @ pts_3d.T + t_w2c).T
    depth = pts_cam[:, 2]  # Z values
    # sort from far (large Z) to near (small Z)
    sort_idx = np.argsort(-depth)
    sorted_pts_3d = list(pts_3d[sort_idx])
    sorted_pts_color = list(pts_color[sort_idx])
    sorted_depth = depth[sort_idx]
    neg_idx = -1
    for i in range(len(sorted_pts_3d)):
        if sorted_depth[i] < 0:  # modified to depth
            neg_idx = i
            break
    if neg_idx != -1:
        sorted_pts_3d = sorted_pts_3d[0:neg_idx]
        sorted_pts_color = sorted_pts_color[0:neg_idx]
    return sorted_pts_3d, sorted_pts_color

if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")  ## from 3D model
    point_desc_df = pd.read_pickle("data/point_desc.pkl")  ## from 2D images

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())  # 3D pos
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)  # 3D descriptors

    # print(f"dexc_df: {desc_df.shape}\n{desc_df.keys}")
    # for str in desc_df.keys():
    #     print(f"{str}: {desc_df[str].shape}")
    # print(f"kp_model: {kp_model.shape}\n{kp_model}")  # X, Y, Z
    # print(f"desc_model: {desc_model.shape}\n{desc_model}") # dim = 128 with n(point) = 111519

    IMAGE_ID_LIST = list(range(1, 294)) # 1..293 (1,294)
    IMAGE_ID_LIST = []
    cnt_img = 0
    cnt_vimg = 0
    for index, row in images_df.iterrows():
        cnt_img += 1
        if row["NAME"] and row["NAME"][0] == 'v':
            IMAGE_ID_LIST.append(row["IMAGE_ID"])
            cnt_vimg += 1
    # IMAGE_ID_LIST = IMAGE_ID_LIST[30:40]
    print(f"~~~ n(img): {cnt_img}, n(vimg): {cnt_vimg}")
    # IMAGE_ID_LIST = [200, 201,202] ### ? // 200, 201
    # IMAGE_ID_LIST = [200]
    
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    for idx in tqdm(IMAGE_ID_LIST):
        # Load quaery image
        # fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        # rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[(point_desc_df["IMAGE_ID"] == idx) & (point_desc_df["POINT_ID"] != -1)]  ## get the match points
        kp_query = np.array(points["XY"].to_list())
        # corres_3d_query = np.array(points["POINT_ID"].to_list())  ##
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
        # print(f"kp_query: {kp_query.shape}\n{kp_query}")
        # print(f"desc_query: {desc_query.shape}\n{desc_query}")

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))   # (2D, 3D)
        # retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query, corres_3d_query), (kp_model, desc_model))   # (2D, 3D)
        Rot_matrix, _ = cv2.Rodrigues(rvec)
        rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
        # print("camera posture:\n This is Rotation_matrix:")  # p = K[R|t_vec]w
        # print(type(Rot_matrix), Rot_matrix)
        # print("This is Rotation_quaternion:")
        # print(type(rotq), rotq)
        tvec = tvec.reshape(1,3)  # tvec = tvec.T
        # print("This is t:")
        # print(type(tvec), tvec)
        # rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
        # tvec = tvec.reshape(1,3) # Reshape translation vector
        r_list.append(rvec)
        t_list.append(tvec)
        
        rotq = rotq[0]
        tvec = tvec[0]
        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = (ground_truth[["QX","QY","QZ","QW"]].values)[0]
        tvec_gt = (ground_truth[["TX","TY","TZ"]].values)[0]
        # print("HERE!!!", rotq_gt, tvec_gt, "!!!")

        # Calculate error
        r_error = rotation_error(rotq, rotq_gt)  ## quaternion represents the diff rotation  //////////////////// modified
        t_error = translation_error(tvec, tvec_gt)
        # print(f"r_error: \n axis:{r_error[0:3]}, radians:{r_error[3]}\nt_error: {t_error}")
        rotation_error_list.append(r_error[3])
        translation_error_list.append(t_error)

    # TODO: calculate median of relative rotation angle differences and translation differences and print them
    print(" ~~ median of ~~ :")
    print(f"angle(rad): {np.median(np.array(rotation_error_list))}")
    print(f"translation: {np.median(np.array(translation_error_list))}")

    # TODO: result visualization
    # Camera2World_Transform_Matrixs = []
    # for r, t in zip(r_list, t_list):   ## for i in range(len(r_list)): r= t=
    #     # TODO: calculate camera pose in world coordinate system  # C = - R.T * t  if [R|t]
    #     c2w = np.eye(4)
    #     Camera2World_Transform_Matrixs.append(c2w)
    # visualization(Camera2World_Transform_Matrixs, points3D_df)
    
    
    Camera2World_Transform_Matrixs = []  ## [[R.T, -(R.T)*t](3*4), [0, 1]] = [[R.T, C], [0, 1]]
    for rvec, tvec in zip(r_list, t_list): # all pairs (rvec, tvec) in order
        R_mat, _ = cv2.Rodrigues(rvec)  # rotation matrix
        tvec = np.asarray(tvec).reshape(3, 1)
        C = (-R_mat.T @ tvec).flatten()
        c2w = np.eye(4) #I(4*4)
        c2w[:3, :3] = R_mat.T
        c2w[:3, 3] = C  # c2w is a c_to_w_matrix(4*4)
        Camera2World_Transform_Matrixs.append(c2w)
        
    save_path_c2w = os.path.join(os.getcwd(), "c2w_matrixs.npy")    ################################
    np.save(save_path_c2w, np.array(Camera2World_Transform_Matrixs))
    # Camera2World_Transform_Matrixs = list(np.load("c2w_matrixs.npy"))  ################################
    
    os.makedirs("data/video_materials", exist_ok=True)
    # virtual_pts_3d = [(1, 2, 3)]  # video-materials
    virtual_pts_3d = np.load("cube_transformed_vertices.npy")
    virtual_pts_color = (np.load("cube_color.npy") * 255)
    # print("~~~~~", virtual_pts_3d, "~~~~~~")
    # virtual_pts_3d = vertex2all(virtual_pts_3d)
    # virtual_pts_color = [(255, 0, 0) for _ in range(len(virtual_pts_3d))]
    # print("!!! the len of vpts: ", len(virtual_pts_3d))
    # state = 0 # to prevent the mistake cases
    iteration = 0
    for idx, c2w in tqdm(zip(IMAGE_ID_LIST, Camera2World_Transform_Matrixs), total=len(IMAGE_ID_LIST)):
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread(f"data/frames/{fname}", cv2.IMREAD_COLOR)
        virtual_pts_3d, virtual_pts_color = sort_depth(c2w, virtual_pts_3d, virtual_pts_color)
        # print("@@@@@@@", virtual_pts_3d, "@@@@@@@")
        h, w = rimg.shape[:2]
        cnt_inlier = 0
        for pt_3d, color in zip(virtual_pts_3d, virtual_pts_color):
            if iteration >= 72:
                break
            # print("@@@@@@@", color, type(color), "@@@@@@@")
            color = tuple(map(int, np.array(color).flatten()))
            p_img = world2camera2image(c2w, [pt_3d])
            p_img = p_img.reshape(-1, 2)
            for (x, y) in p_img.astype(int):
                if 0 <= x < w and 0 <= y < h: # and state < 2:
                    cv2.circle(rimg, (x, y), 8, color, -1)
                    cnt_inlier += 1
                # else:
                    # print(f"Point {(x, y)} out of image bounds for {fname}")
            # if (cnt_inlier != 0 and state == 0) or (cnt_inlier == 0 and state == 1):
            #     state += 1
            #     print(f"img_idx is {iteration}")
        iteration += 1
        out_path = f"data/video_materials/{fname}"
        cv2.imwrite(out_path, rimg)

    visualization(Camera2World_Transform_Matrixs, points3D_df, scale=0.2)  #0.3
