from pathlib import Path


def check_generation_paths(scene_root, output_root):
    """校验对象: generate_bev_grids 的输入/输出目录 —— 源场景存在且输出不覆盖源目录。"""
    scene_root = Path(scene_root).resolve()
    output_root = Path(output_root).resolve()
    if not scene_root.is_dir():
        raise FileNotFoundError("BEV 栅格源场景目录不存在：{}".format(scene_root))
    if output_root == scene_root or scene_root in output_root.parents:
        raise ValueError("BEV 栅格输出目录不能位于源场景目录内部：{}".format(output_root))


def check_source_scene(scene_dir):
    """校验对象: 单场景输入 —— 必须包含可读的 LMDB 数据文件。"""
    lmdb_dir = Path(scene_dir) / "lmdb"
    if not (lmdb_dir / "data.mdb").is_file():
        raise FileNotFoundError("场景缺少 LMDB：{}".format(lmdb_dir))
