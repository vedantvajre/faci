"""
Sentiment Analysis Task (SST-2)
"""

from typing import Dict, List, Optional, Any

from mechinterp.core.base import Architecture
from mechinterp.core.registry import register_task
from mechinterp.tasks.base import ClassificationTask


@register_task("sentiment")
class SentimentTask(ClassificationTask):
    """
    Sentiment classification using SST-2 dataset.
    
    Binary classification: positive (1) vs negative (0)
    """

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="nyu-mll/glue",
            dataset_config="sst2",
            text_column="sentence",
            label_column="label",
            positive_label=1,  # positive sentiment
            negative_label=0,  # negative sentiment
            **kwargs
        )
        
        # Templates for different architectures
        self._templates = {
            Architecture.DECODER: '{text}\nSentiment:',
            Architecture.ENCODER_DECODER: 'sentiment: {text}',
        }

    @property
    def name(self) -> str:
        return "sentiment"

    def load_data(
        self,
        split: str = "validation",  # SST-2 test set has no labels
        max_samples: Optional[int] = None,
    ) -> None:
        """Load SST-2 data."""
        super().load_data(split=split, max_samples=max_samples)

    def _format_for_decoder(self, text: str) -> str:
        """Format for decoder: text + sentiment prompt."""
        return f"{text}\nSentiment:"

    def _format_for_enc_dec(self, text: str) -> str:
        """Format for encoder-decoder: task prefix."""
        return f"classify sentiment: {text}"

    def get_target_tokens(self, architecture: Architecture) -> Dict[str, str]:
        """
        Get target tokens for each class.
        
        Useful for decoder models where we check which token has higher prob.
        """
        if architecture == Architecture.DECODER:
            return {
                "positive": " positive",
                "negative": " negative",
            }
        elif architecture == Architecture.ENCODER_DECODER:
            return {
                "positive": "positive",
                "negative": "negative",
            }
        return {}


@register_task("sentiment-imdb")
class SentimentIMDBTask(ClassificationTask):
    """
    Alternative sentiment task using IMDB dataset.
    
    Longer movie reviews, good for testing on different text lengths.
    """

    def __init__(self, max_length: int = 512, **kwargs):
        super().__init__(
            dataset_name="imdb",
            dataset_config=None,
            text_column="text",
            label_column="label",
            positive_label=1,
            negative_label=0,
            max_length=max_length,
            **kwargs
        )

    @property
    def name(self) -> str:
        return "sentiment-imdb"

    def _format_for_decoder(self, text: str) -> str:
        # Truncate long reviews for decoder
        text = text[:1000] if len(text) > 1000 else text
        return f"Review: {text}\nSentiment:"

    def _format_for_enc_dec(self, text: str) -> str:
        text = text[:1000] if len(text) > 1000 else text
        return f"classify sentiment: {text}"
