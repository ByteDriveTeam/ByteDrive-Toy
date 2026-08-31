def check_world_model_train_inputs(model, loader, optimizer):
    """校验对象: train_world_model_epoch 入参 —— 训练对象必须有效。"""
    if model is None or loader is None or optimizer is None:
        raise TypeError("世界模型、DataLoader、优化器均不能为空")
