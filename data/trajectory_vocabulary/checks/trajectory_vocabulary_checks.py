"""轨迹词表输入校验。"""

def check_vocab_array(value, name):
    """校验对象: 词表数组 —— 必须为有限二维或三维数值数组。"""
    if value.ndim not in (2, 3) or not value.size:
        raise ValueError("{} 词表数组为空或维度错误".format(name))
