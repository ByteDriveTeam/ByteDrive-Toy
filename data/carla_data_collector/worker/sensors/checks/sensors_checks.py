# 本文件为 worker/sensors/sensors.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_ego(ego):
    """校验对象: SensorRig 构造入参 ego —— 传感器需 attach 到有效主车 actor。"""
    assert ego is not None, "SensorRig 需要一个有效的主车 actor 作为挂载对象"


def check_no_future_frame(got_frame, expected_frame):
    """校验对象: gather 取到的传感器帧号 —— 不允许超前于本次 tick 的帧号。

    超前意味着传感器跑到了仿真前面，严格同步被破坏，宁可报错也不写入错乱数据。
    """
    assert got_frame <= expected_frame, \
        "传感器帧 {} 超前于期望帧 {}，严格同步被破坏".format(got_frame, expected_frame)
