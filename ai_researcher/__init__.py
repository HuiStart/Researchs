from .cycle_researcher import CycleResearcher
from .cycle_reviewer import CycleReviewer
from .deep_reviewer import DeepReviewer
from .detector import AIDetector

'''
导出核心功能类：
    CycleResearcher（论文生成）、
    CycleReviewer（论文评审）、
    DeepReviewer（深度评审）、
    AIDetector（AI 生成检测）
'''
__all__ = ['CycleResearcher', 'CycleReviewer', 'DeepReviewer', 'AIDetector']

__version__ = '0.1.0'