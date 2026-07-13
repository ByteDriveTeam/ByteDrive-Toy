# 本文件为 model/frustum_encoding/frustum_encoding.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_frustum_args(out_dim, patch_size, mlp_hidden, coord_symlog_scale):
    """校验对象: FrustumEncoding 构造入参 —— 维度/patch/隐藏维为正、symlog 尺度为正。"""
    if out_dim < 1 or patch_size < 1 or mlp_hidden < 1:
        raise ValueError("out_dim/patch_size/mlp_hidden 必须为正整数，实际 {}/{}/{}。".format(
            out_dim, patch_size, mlp_hidden))
    if coord_symlog_scale <= 0:
        raise ValueError("coord_symlog_scale 必须为正数，实际为 {}。".format(coord_symlog_scale))


def check_frustum_inputs(patch_features, intrinsics, extrinsics, patch_size):
    """校验对象: FrustumEncoding.forward 入参 —— 特征 [B,C,gh,gw]、内参 [B,4]、外参 [B,6]，批次一致。"""
    if patch_features.ndim != 4:
        raise ValueError("patch_features 期望 (B,C,gh,gw) 四维，实际 {}。".format(tuple(patch_features.shape)))
    b = int(patch_features.shape[0])
    if intrinsics.shape != (b, 4):
        raise ValueError("intrinsics 期望 ({},4)（fx,fy,cx,cy），实际 {}。".format(b, tuple(intrinsics.shape)))
    if extrinsics.shape != (b, 6):
        raise ValueError("extrinsics 期望 ({},6)（x,y,z,roll,pitch,yaw），实际 {}。".format(b, tuple(extrinsics.shape)))
