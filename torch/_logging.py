"""
torch._logging -- 日志系统桩

对应 PyTorch: torch._logging

提供最小化的兼容实现，支持 set_logs(graph_code=True) 等日志开关。
"""

_log_settings = {}


def set_logs(**kwargs):
    """设置日志选项"""
    _log_settings.update(kwargs)


def get_log_settings():
    """获取当前日志设置"""
    return _log_settings
