# 本文件为 common/protocol/protocol.py 的校验伴随文件（规范 §7.1，免文件头）。

_ALLOWED_CMDS = {"init", "query_spawn_points", "start_scene", "continue_scene", "shutdown"}


def check_command(msg):
    """校验对象: protocol 命令消息 msg —— 必须含合法 cmd 与 dict 形式的 args。"""
    assert isinstance(msg, dict), "命令消息必须是 dict"
    assert msg.get("cmd") in _ALLOWED_CMDS, "未知命令 cmd={!r}".format(msg.get("cmd"))
    assert isinstance(msg.get("args", {}), dict), "命令 args 必须是 dict"


def check_response(msg):
    """校验对象: protocol 响应消息 msg —— 必须含 bool ok；失败时须带 error。"""
    assert isinstance(msg, dict), "响应消息必须是 dict"
    assert isinstance(msg.get("ok"), bool), "响应必须含 bool 字段 ok"
    if not msg["ok"]:
        assert msg.get("error"), "失败响应必须带 error 描述"
