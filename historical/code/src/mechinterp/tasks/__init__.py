"""
Task implementations for mechanistic interpretability.
"""

from mechinterp.tasks.base import ClassificationTask, SpanExtractionTask, GenerationTask
from mechinterp.tasks.sentiment import SentimentTask
from mechinterp.tasks.nli import NLITask
from mechinterp.tasks.qa import QATask
from mechinterp.tasks.factual_recall import FactualRecallTask

__all__ = [
    "ClassificationTask",
    "SpanExtractionTask", 
    "GenerationTask",
    "SentimentTask",
    "NLITask",
    "QATask",
    "FactualRecallTask",
]
