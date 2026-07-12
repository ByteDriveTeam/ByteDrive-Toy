# 本文件为 common/shm/shm.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_arena_args(name, size_bytes):
    """校验对象: Arena 构造入参 name/size_bytes —— 名字非空、容量为正整数。"""
    assert isinstance(name, str) and name, "arena name 必须是非空字符串"
    assert isinstance(size_bytes, int) and size_bytes > 0, "arena size_bytes 必须是正整数"


def check_put(data):
    """校验对象: BumpAllocator.put 入参 data —— 必须是非空、支持 buffer 协议的字节。"""
    assert isinstance(data, (bytes, bytearray, memoryview)), "put 仅接受 bytes/bytearray/memoryview"
    assert len(data) > 0, "put 不接受空数据块"
