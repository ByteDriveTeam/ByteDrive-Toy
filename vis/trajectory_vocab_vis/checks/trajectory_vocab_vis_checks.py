"""词表可视化输入校验。"""

def check_bundle(bundle):
    """校验对象: 词表对象字典 —— 必须包含 vocab 与 metrics。"""
    if not isinstance(bundle, dict) or "vocab" not in bundle or "metrics" not in bundle:
        raise ValueError("词表对象字典缺少 vocab/metrics")
