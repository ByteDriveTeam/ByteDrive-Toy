def check_world_model_train_inputs(model, loader, optimizer, stage):
    """校验对象: train_world_model_epoch 入参 —— 训练对象有效且阶段至少启用一种目标。"""
    if model is None or loader is None or optimizer is None:
        raise TypeError("世界模型、DataLoader、优化器均不能为空")
    if stage.reconstruction_weight <= 0 and stage.visreg_weight <= 0:
        raise ValueError("训练阶段必须启用掩码补全或 VISReg 中的至少一项")
