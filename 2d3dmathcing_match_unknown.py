from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time

from tqdm import tqdm

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
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27, 0, 540],
                             [0, 1869.18, 960],
                             [0, 0, 1]], dtype=np.float32)
    distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352], dtype=np.float32)

    # --- Step 1: 建立匹配器並進行 knn 比對 (k=2)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc_query, desc_model, k=2)
    print(f"matches: {type(matches)}")

    # --- Step 2: Ratio Test (Lowe’s ratio test)
    good_matches = []
    for m, n in matches:  # 2nd good is for removing faked match
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

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
    q1 = q1 / len_q1
    q2 = q2 / len_q2  ## normalized
    
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

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    #TODO: visualize the camera pose
    pass

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

    # IMAGE_ID_LIST = [200]  ##### ? // 200, 201
    IMAGE_ID_LIST = list(range(1, 294)) # 1..293
    
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    for idx in tqdm(IMAGE_ID_LIST):
        # Load quaery image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]  ## get the match points
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
        # print(f"kp_query: {kp_query.shape}\n{kp_query}")
        # print(f"desc_query: {desc_query.shape}\n{desc_query}")

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))   # (2D, 3D)
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
    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        # TODO: calculate camera pose in world coordinate system  # C = - R.T * t  if [R|t]
        c2w = np.eye(4)
        Camera2World_Transform_Matrixs.append(c2w)
    visualization(Camera2World_Transform_Matrixs, points3D_df)