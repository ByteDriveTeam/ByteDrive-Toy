# 本文件为 worker/actors.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_blueprints(blueprints, filter_str):
    """校验对象: spawn_* 用的蓝图过滤结果 —— 过滤器必须命中至少一个蓝图。

    命中为空说明 config 里的 filter 与当前 Carla 资产不符，应尽早失败而非静默生成 0 个 actor。
    """
    assert len(blueprints) > 0, "蓝图过滤器未命中任何资产: {!r}".format(filter_str)
