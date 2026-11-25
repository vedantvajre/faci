"""
Base Task Classes for Mechanistic Interpretability

Provides ClassificationTask and SpanExtractionTask base classes
that specific task implementations inherit from.
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Any

from mechinterp.core.base import Architecture, BaseTask


class ClassificationTask(BaseTask):
    """
    Base class for binary/multi-class classification tasks.
    
    Subclasses should implement:
        - name property
        - _format_for_decoder (optional)
        - _format_for_enc_dec (optional)
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        text_column: str = "text",
        label_column: str = "label",
        positive_label: int = 1,
        negative_label: int = 0,
        max_length: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration/subset
            text_column: Column containing input text
            label_column: Column containing labels
            positive_label: Label value for positive class
            negative_label: Label value for negative class
            max_length: Maximum sequence length (optional)
        """
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_column = text_column
        self.label_column = label_column
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.max_length = max_length
        self._data = None
        self._templates = {}

    @property
    def num_classes(self) -> int:
        """Number of classes (default: binary)."""
        return 2

    def load_data(
        self,
        split: str = "validation",
        max_samples: Optional[int] = None,
    ) -> None:
        """Load dataset from HuggingFace."""
        from datasets import load_dataset
        
        if self.dataset_config:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=split,
            )
        else:
            dataset = load_dataset(self.dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self._data = dataset

    def get_positive_examples(
        self, 
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Get examples with positive label.
        
        Args:
            n: Number of examples to return
            seed: Random seed for reproducibility - shuffles examples before selection
        """
        import random
            
        if self._data is None:
            self.load_data()
        
        examples = [
            self._get_text(ex)
            for ex in self._data
            if ex[self.label_column] == self.positive_label
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
        """
        Get examples with negative label.
        
        Args:
            n: Number of examples to return
            seed: Random seed for reproducibility - shuffles examples before selection
        """
        import random
            
        if self._data is None:
            self.load_data()
        
        examples = [
            self._get_text(ex)
            for ex in self._data
            if ex[self.label_column] == self.negative_label
        ]
        
        # Shuffle with seed for reproducible but varied sampling
        if seed is not None:
            random.seed(seed)
            random.shuffle(examples)
        
        if n:
            examples = examples[:n]
        return examples

    def _get_text(self, example: Dict[str, Any]) -> str:
        """Extract text from example. Override for complex cases."""
        return example[self.text_column]

    def format_for_architecture(
        self,
        text: str,
        architecture: Architecture,
    ) -> str:
        """Format text for specific architecture."""
        if architecture == Architecture.ENCODER:
            return self._format_for_encoder(text)
        elif architecture == Architecture.DECODER:
            return self._format_for_decoder(text)
        elif architecture == Architecture.ENCODER_DECODER:
            return self._format_for_enc_dec(text)
        return text

    def format_input(
        self,
        example: Dict[str, Any],
        architecture: Architecture,
    ) -> str:
        """Format a full example for the given architecture."""
        text = self._get_text(example)
        return self.format_for_architecture(text, architecture)

    def _format_for_encoder(self, text: str) -> str:
        """Format for encoder models. Override as needed."""
        return text

    def _format_for_decoder(self, text: str) -> str:
        """Format for decoder models. Override as needed."""
        return text

    def _format_for_enc_dec(self, text: str) -> str:
        """Format for encoder-decoder models. Override as needed."""
        return text

    def get_label(self, example: Dict[str, Any]) -> int:
        """Get label from example."""
        return example[self.label_column]

    def get_target_tokens(self, architecture: Architecture) -> Dict[str, str]:
        """Get target tokens for each class. Override in subclass."""
        return {}

    def evaluate(
        self,
        predictions: List[int],
        labels: List[int],
    ) -> Dict[str, float]:
        """Evaluate classification performance."""
        correct = sum(p == l for p, l in zip(predictions, labels))
        n = len(predictions)
        return {
            "accuracy": correct / n if n > 0 else 0,
            "n_samples": n,
        }


class SpanExtractionTask(BaseTask):
    """
    Base class for span extraction / QA tasks.
    
    Subclasses should implement:
        - name property
        - format_input (optional)
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        context_column: str = "context",
        question_column: str = "question",
        answer_column: str = "answers",
        **kwargs
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration/subset
            context_column: Column containing context/passage
            question_column: Column containing question
            answer_column: Column containing answer(s)
        """
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.context_column = context_column
        self.question_column = question_column
        self.answer_column = answer_column
        self._data = None

    @property
    def num_classes(self) -> int:
        """Not applicable for span extraction."""
        return -1

    def load_data(
        self,
        split: str = "validation",
        max_samples: Optional[int] = None,
    ) -> None:
        """Load dataset from HuggingFace."""
        from datasets import load_dataset
        
        if self.dataset_config:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=split,
            )
        else:
            dataset = load_dataset(self.dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self._data = dataset

    def _get_answer(self, example: Dict[str, Any]) -> str:
        """
        Extract answer text from example.
        
        Handles SQuAD-style answers format.
        """
        answers = example.get(self.answer_column, {})
        
        # SQuAD format: {"text": [...], "answer_start": [...]}
        if isinstance(answers, dict):
            texts = answers.get("text", [])
            if texts and len(texts) > 0:
                return texts[0]
        
        # Simple string format
        if isinstance(answers, str):
            return answers
        
        # List format
        if isinstance(answers, list) and len(answers) > 0:
            return answers[0] if isinstance(answers[0], str) else ""
        
        return ""

    def get_examples(self, n: Optional[int] = None) -> List[Dict]:
        """
        Get processed examples with context, question, answer.
        """
        if self._data is None:
            self.load_data()
        
        examples = []
        for ex in self._data:
            examples.append({
                "context": ex[self.context_column],
                "question": ex[self.question_column],
                "answer": self._get_answer(ex),
            })
        
        if n:
            examples = examples[:n]
        return examples

    def format_input(
        self,
        example: Dict[str, Any],
        architecture: Architecture,
    ) -> str:
        """Format QA input for architecture."""
        context = example.get(self.context_column, example.get("context", ""))
        question = example.get(self.question_column, example.get("question", ""))
        
        if architecture == Architecture.ENCODER:
            return f"{question} [SEP] {context}"
        elif architecture == Architecture.DECODER:
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        elif architecture == Architecture.ENCODER_DECODER:
            return f"question: {question} context: {context}"
        return f"{question} {context}"

    def get_label(self, example: Dict[str, Any]) -> int:
        """Not applicable for span extraction."""
        return -1

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate span extraction.
        
        Uses exact match and F1 score.
        """
        exact = 0
        f1_total = 0
        
        for pred, ref in zip(predictions, references):
            pred_normalized = pred.lower().strip()
            ref_normalized = ref.lower().strip()
            
            # Exact match
            if pred_normalized == ref_normalized:
                exact += 1
            
            # Token-level F1
            pred_tokens = set(pred_normalized.split())
            ref_tokens = set(ref_normalized.split())
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1 = 1.0 if pred_tokens == ref_tokens else 0.0
            else:
                common = pred_tokens & ref_tokens
                precision = len(common) / len(pred_tokens) if pred_tokens else 0
                recall = len(common) / len(ref_tokens) if ref_tokens else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_total += f1
        
        n = len(predictions)
        return {
            "exact_match": exact / n if n > 0 else 0,
            "f1": f1_total / n if n > 0 else 0,
            "n_samples": n,
        }


class GenerationTask(BaseTask):
    """
    Base class for text generation tasks.
    
    For tasks like summarization, translation, etc.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        source_column: str = "source",
        target_column: str = "target",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.source_column = source_column
        self.target_column = target_column
        self._data = None

    @property
    def num_classes(self) -> int:
        """Not applicable for generation."""
        return -1

    def load_data(
        self,
        split: str = "validation",
        max_samples: Optional[int] = None,
    ) -> None:
        """Load dataset from HuggingFace."""
        from datasets import load_dataset
        
        if self.dataset_config:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=split,
            )
        else:
            dataset = load_dataset(self.dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self._data = dataset

    def format_input(
        self,
        example: Dict[str, Any],
        architecture: Architecture,
    ) -> str:
        """Format generation input."""
        source = example[self.source_column]
        
        if architecture == Architecture.DECODER:
            return source
        elif architecture == Architecture.ENCODER_DECODER:
            return source
        return source

    def get_target(self, example: Dict[str, Any]) -> str:
        """Get target text."""
        return example[self.target_column]

    def get_label(self, example: Dict[str, Any]) -> int:
        """Not applicable."""
        return -1
