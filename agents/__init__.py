"""
Multi-Agent Radiology Report Classification System

This package contains four specialized agents for analyzing radiology reports:
- Tagger: Extracts metadata (modality, body region, normality)
- Classifier: Identifies disease type and generates clinical impression
- Recommender: Generates follow-up recommendations
- Orchestrator: Evaluates outputs and provides feedback for iterative refinement
"""

from .tagger import tag_report, Tags
from .classifier import classify_report, Classification
from .recommender import recommend_follow_up, Recommendation
from .orchestrator import evaluate_report, RadiologistEvaluation

__all__ = [
    'tag_report',
    'Tags',
    'classify_report',
    'Classification',
    'recommend_follow_up',
    'Recommendation',
    'evaluate_report',
    'RadiologistEvaluation',
]
