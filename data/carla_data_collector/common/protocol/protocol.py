"""控制管道协议：Py312 collector 与 Py37 worker 间的 JSON 行命令/响应。

模块: common/protocol/protocol.py
依赖: json
读取配置: —（仅定义消息格式，不读 config 键）
对外接口:
    - 命令常量: CMD_INIT / CMD_QUERY_SPAWN_POINTS / CMD_START_SCENE / CMD_CONTINUE_SCENE / CMD_SHUTDOWN
    - 状态常量: STATUS_OK / STATUS_MAX_FRAMES / STATUS_PARTIAL / STATUS_COLLISION / STATUS_UNREACHABLE
    - make_command(cmd, **args) -> dict
    - make_response(ok, result=None, error=None) -> dict
    - write_message(stream, obj) -> None     # 向二进制流写一行 JSON 并 flush
    - read_message(stream) -> dict | None     # 读一行 JSON；EOF 返回 None
说明: 控制面走 stdin/stdout 二进制管道，每条消息一行 UTF-8 JSON。约定 worker 的 stdout
      只承载本协议，所有日志/异常打印一律走 stderr，避免污染消息流（见 worker/main.py）。
      大块传感器数据不走本协议，而走共享内存（common/shm.py）；本协议只传帧索引等元数据。
"""

import json


class ProtocolError(RuntimeError):
    """协议流异常：读到非空但无法解析为 JSON 的行（通常是 stdout 被零散输出污染）。"""


# 命令（collector -> worker）
CMD_INIT = "init"                          # 下发配置 + arena 信息，连接 Carla
CMD_QUERY_SPAWN_POINTS = "query_spawn_points"  # 加载地图并返回全部可达点（用于建路线队列）
CMD_START_SCENE = "start_scene"            # 重载地图→布景→预热→采集首段（一次行驶的开端）
CMD_CONTINUE_SCENE = "continue_scene"      # 复用存活世界续采下一段（arena 已被 collector 读空）
CMD_SHUTDOWN = "shutdown"                  # 退出 worker 进程

# 语义 Lidar 点的结构化 dtype（两端用 numpy.dtype(...) 还原，避免重复定义）。
# 对应 carla 语义雷达每点字段：坐标、入射角余弦、命中物体索引与语义标签。
SEMANTIC_LIDAR_DTYPE = [
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("cos_angle", "<f4"), ("obj_idx", "<u4"), ("obj_tag", "<u4"),
]

# 帧索引里大块张量的 blob 字段键（offset/size 指向共享内存 arena）。
BLOB_OFFSET = "offset"
BLOB_SIZE = "size"
BLOB_SHAPE = "shape"
BLOB_DTYPE = "dtype"

# start_scene / continue_scene 的段状态（worker -> collector）
STATUS_OK = "ok"                           # 到达终点：本段为该次行驶的最后一段
STATUS_MAX_FRAMES = "max_frames"           # 达整次行驶总帧上限：本段为最后一段
STATUS_PARTIAL = "partial"                 # arena 写满但未结束：本段已满，行驶继续（续采下一段）
STATUS_COLLISION = "collision"             # 主车碰撞：当前未落盘段丢弃，行驶结束
STATUS_UNREACHABLE = "route_unreachable"   # 规划不出到目标点的路线（仅 start_scene）


def make_command(cmd, **args):
    """构造一条命令消息。"""
    return {"cmd": cmd, "args": args}


def make_response(ok, result=None, error=None):
    """构造一条响应消息。"""
    return {"ok": bool(ok), "result": result if result is not None else {}, "error": error}


def write_message(stream, obj):
    """向二进制流写入一行 UTF-8 JSON 并立即 flush。"""
    line = json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n"
    stream.write(line)
    stream.flush()


def read_message(stream):
    """从二进制流读取一行 JSON；流结束(EOF)返回 None。

    读到非空但非 JSON 的行（消息流被零散 stdout 输出污染）时，抛出带原始内容的
    ProtocolError，而非裸 JSONDecodeError，便于一眼定位污染来源。
    """
    line = stream.readline()
    if not line:
        return None
    text = line.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            "消息流被非协议输出污染，无法解析为 JSON：{!r}（{}）".format(text[:500], exc))
