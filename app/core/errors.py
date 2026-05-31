"""异常类型与稳定性处理.

目标:
 - MCAP 不存在 / 损坏 / 未知 Topic 不能崩溃
 - 单帧解码失败不影响后续帧;单个 MCAP 失败不影响其他 MCAP
 - 输出目录不存在时自动创建;磁盘/模型加载失败有明确错误
 - API 非法参数不导致服务崩溃;Ctrl+C 可优雅退出
"""
class PipelineError(Exception):
    """管线基础异常类型。"""


class McapReadError(PipelineError):
    """MCAP 读取失败。"""


class DecodeError(PipelineError):
    """图像解码失败。"""


class ModelLoadError(PipelineError):
    """模型加载失败。"""
