"""
Generic Multi-Agent Framework
"""

from .agent import GenericAgent
try:
    from .local_model import LocalModel
except ImportError:
    LocalModel = None

__all__ = [
    'GenericAgent',
    'LocalModel',
]
