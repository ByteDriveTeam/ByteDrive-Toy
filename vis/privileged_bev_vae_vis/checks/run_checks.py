"""特权 BEV VAE 可视化命令行输入检查。

模块: vis/privileged_bev_vae_vis/checks/run_checks.py
依赖: 无
读取配置: 无
对外接口:
    - check_frame(frame, num_frames) -> None
"""


def check_frame(frame, num_frames):
    """校验对象: CLI 选择的场景帧编号。"""
    if frame < 0 or frame >= num_frames:
        raise ValueError("frame 必须位于 [0, {})".format(num_frames))
