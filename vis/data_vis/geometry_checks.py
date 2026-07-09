import numpy as np


# 校验对象: transform_matrix 的入参 pose6 —— 必须是 6 元位姿 [x,y,z,roll,pitch,yaw]
def check_pose6(pose6):
    assert len(pose6) == 6, "pose6 期望 [x,y,z,roll,pitch,yaw] 六元，得到长度 {}".format(len(pose6))


# 校验对象: project_points 的入参 pts_world —— 必须是 (N,3) 三维点集
def check_points(pts):
    arr = np.asarray(pts)
    assert arr.ndim == 2 and arr.shape[1] == 3, "点集期望 (N,3)，得到 {}".format(arr.shape)
