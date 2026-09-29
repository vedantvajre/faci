"""
Natural Language Inference Task (MNLI/SNLI)
"""

from typing import Dict, List, Optional, Any

from mechinterp.core.base import Architecture
from mechinterp.core.registry import register_task
from mechinterp.tasks.base import ClassificationTask


@register_task("nli")
class NLITask(ClassificationTask):
    """
    Natural Language Inference using MNLI dataset.
    
    Three-way classification: entailment (0), neutral (1), contradiction (2)
    For binary analysis, we use entailment vs contradiction.
    """

    def __init__(self, binary: bool = True, **kwargs):
        """
        Args:
            binary: If True, use only entailment (pos) and contradiction (neg)
        """
        super().__init__(
            dataset_name="nyu-mll/glue",
            dataset_config="mnli",
            text_column="premise",  # Will combine with hypothesis
            label_column="label",
            positive_label=0,  # entailment
            negative_label=2,  # contradiction
            **kwargs
        )
        self.binary = binary
        self._templates = {
            Architecture.DECODER: 'Premise: {premise}\nHypothesis: {hypothesis}\nRelation:',
            Architecture.ENCODER_DECODER: 'nli premise: {premise} hypothesis: {hypothesis}',
        }

    @property
    def name(self) -> str:
        return "nli"

    @property
    def num_classes(self) -> int:
        return 2 if self.binary else 3

    def load_data(
        self,
        split: str = "validation_matched",
        max_samples: Optional[int] = None,
    ) -> None:
        """Load MNLI data."""
        super().load_data(split=split, max_samples=max_samples)
        
        # Filter to binary if needed
        if self.binary and self._data is not None:
            self._data = self._data.filter(
                lambda x: x["label"] in [self.positive_label, self.negative_label]
            )

    def _get_text(self, example: Dict[str, Any]) -> str:
        """Combine premise and hypothesis."""
        return f"{example['premise']} [SEP] {example['hypothesis']}"

    def get_positive_examples(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Get entailment examples."""
        import random
            
        if self._data is None:
            self.load_data()
        
        examples = [
            self._get_text(ex)
            for ex in self._data
            if ex["label"] == self.positive_label
        ]
        
        # Shuffle with seed for reproducible but varied sampling
        if seed is not None:
            random.seed(seed)
            random.shuffle(examples)
        
        if n:
            examples = examples[:n]
        return examples

    def get_negative_examples(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Get contradiction examples."""
        import random
            
        if self._data is None:
            self.load_data()
        
        examples = [
            self._get_text(ex)
            for ex in self._data
            if ex["label"] == self.negative_label
        ]
        
        # Shuffle with seed for reproducible but varied sampling
        if seed is not None:
            random.seed(seed)
            random.shuffle(examples)
        
        if n:
            examples = examples[:n]
        return examples

    def format_input(
        self,
        example: Dict[str, Any],
        architecture: Architecture,
    ) -> str:
        """Format NLI input for architecture."""
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        
        if architecture == Architecture.ENCODER:
            return f"{premise} [SEP] {hypothesis}"
        elif architecture == Architecture.DECODER:
            return f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation:"
        elif architecture == Architecture.ENCODER_DECODER:
            return f"nli premise: {premise} hypothesis: {hypothesis}"
        return f"{premise} {hypothesis}"

    def _format_for_encoder(self, text: str) -> str:
        """Already formatted in _get_text."""
        return text

    def _format_for_decoder(self, text: str) -> str:
        """Add prompt for decoder."""
        # Text is already "premise [SEP] hypothesis"
        parts = text.split(" [SEP] ")
        if len(parts) == 2:
            return f"Premise: {parts[0]}\nHypothesis: {parts[1]}\nRelation:"
        return f"{text}\nRelation:"

    def _format_for_enc_dec(self, text: str) -> str:
        """Add task prefix for enc-dec."""
        parts = text.split(" [SEP] ")
        if len(parts) == 2:
            return f"nli premise: {parts[0]} hypothesis: {parts[1]}"
        return f"nli: {text}"

    def get_target_tokens(self, architecture: Architecture) -> Dict[str, str]:
        """Get target tokens for each class."""
        if architecture == Architecture.DECODER:
            return {
                "entailment": " entailment",
                "contradiction": " contradiction",
            }
        elif architecture == Architecture.ENCODER_DECODER:
            return {
                "entailment": "entailment",
                "contradiction": "contradiction",
            }
        return {}


@register_task("snli")
class SNLITask(NLITask):
    """SNLI dataset variant (simpler domain)."""

    def __init__(self, binary: bool = True, **kwargs):
        super().__init__(binary=binary, **kwargs)
        self.dataset_name = "snli"
        self.dataset_config = None

    @property
    def name(self) -> str:
        return "snli"

    def load_data(
        self,
        split: str = "validation",
        max_samples: Optional[int] = None,
    ) -> None:
        """Load SNLI data."""
        from datasets import load_dataset
        
        dataset = load_dataset(self.dataset_name, split=split)
        
        # SNLI has -1 labels for examples without gold label
        dataset = dataset.filter(lambda x: x["label"] >= 0)
        
        if self.binary:
            dataset = dataset.filter(
                lambda x: x["label"] in [self.positive_label, self.negative_label]
            )
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self._data = dataset
