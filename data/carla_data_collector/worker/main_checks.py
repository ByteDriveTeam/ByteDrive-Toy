# 本文件为 worker/main.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_init_args(args):
    """校验对象: init 命令下发的 args —— 必须含 config(dict) 与 arena(name/size_bytes)。

    给出明确报错替代默认 KeyError，便于定位 collector 下发内容不完整的问题。
    （单条命令的 cmd/args 合法性已由 protocol_checks.check_command 把关，此处只查 init 专有字段。）
    """
    config = args.get("config")
    assert isinstance(config, dict) and "carla_collector" in config, \
        "init 缺少有效 config（顶层应含 carla_collector）"
    arena = args.get("arena")
    assert isinstance(arena, dict) and arena.get("name") and isinstance(arena.get("size_bytes"), int), \
        "init 缺少有效 arena（需 name 与整数 size_bytes）"
