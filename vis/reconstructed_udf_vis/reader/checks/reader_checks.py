"""稀疏 TUDF 查看器路径校验。"""

from pathlib import Path


def check_udf_path(path):
    path = Path(path)
    assert path.is_file() and path.name.endswith(".udf.pt"), \
        "TUDF 文件不存在或后缀非法: {}".format(path)

