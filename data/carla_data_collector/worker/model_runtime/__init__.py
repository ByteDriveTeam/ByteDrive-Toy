"""模型闭环采集运行时公开接口。

模块: worker/model_runtime/__init__.py
依赖: worker.model_runtime.model_runtime
读取配置: —
对外接口: ModelCollectionRuntime
"""

from worker.model_runtime.model_runtime import ModelCollectionRuntime

__all__ = ["ModelCollectionRuntime"]
