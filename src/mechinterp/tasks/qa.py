"""
Question Answering Task (SQuAD)

Extractive and generative QA.
"""

from typing import Dict, List, Optional, Any

from mechinterp.core.base import Architecture
from mechinterp.core.registry import register_task
from mechinterp.tasks.base import SpanExtractionTask


@register_task("qa")
class QATask(SpanExtractionTask):
    """
    Question Answering using SQuAD dataset.
    
    Extractive QA: Find answer span in context.
    """

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="rajpurkar/squad",
            dataset_config=None,
            context_column="context",
            question_column="question",
            answer_column="answers",
            **kwargs
        )

    @property
    def name(self) -> str:
        return "qa"

    def load_data(
        self,
        split: str = "validation",
        max_samples: Optional[int] = None,
    ) -> None:
        """Load SQuAD data."""
        super().load_data(split=split, max_samples=max_samples)

    def format_input(
        self,
        example: Dict[str, Any],
        architecture: Architecture,
    ) -> str:
        """Format QA input for architecture."""
        context = example[self.context_column]
        question = example[self.question_column]
        
        # Truncate context if too long
        max_context = 400
        if len(context) > max_context:
            # Try to keep the answer in context
            answer = self._get_answer(example)
            if answer in context:
                idx = context.find(answer)
                start = max(0, idx - max_context // 2)
                context = context[start:start + max_context]
            else:
                context = context[:max_context]
        
        if architecture == Architecture.ENCODER:
            return f"{question} [SEP] {context}"
        elif architecture == Architecture.DECODER:
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        elif architecture == Architecture.ENCODER_DECODER:
            return f"question: {question} context: {context}"
        return f"{question} {context}"

    def get_answerable_examples(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """Get examples that have answers."""
        import random
            
        examples = self.get_examples(n=None)  # Get all first
        answerable = [e for e in examples if e["answer"]]
        
        # Shuffle with seed for reproducible but varied sampling
        if seed is not None:
            random.seed(seed)
            random.shuffle(answerable)
        
        if n:
            answerable = answerable[:n]
        return answerable

    def get_positive_examples(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Get question-context pairs with correct answers."""
        examples = self.get_answerable_examples(n, seed=seed)
        return [
            f"Context: {e['context'][:300]} Question: {e['question']} Answer: {e['answer']}"
            for e in examples
        ]

    def get_negative_examples(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Get question-context pairs with wrong answers (mismatched)."""
        import random
        if seed is not None:
            random.seed(seed)
            
        examples = self.get_answerable_examples(n, seed=seed)
        
        # Shuffle answers to create incorrect pairs
        answers = [e["answer"] for e in examples]
        shuffled = answers.copy()
        random.shuffle(shuffled)
        
        # Ensure no correct matches
        for i in range(len(shuffled)):
            if shuffled[i] == answers[i]:
                j = (i + 1) % len(shuffled)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        
        return [
            f"Context: {e['context'][:300]} Question: {e['question']} Answer: {ans}"
            for e, ans in zip(examples, shuffled)
        ]

    def get_input_for_model(
        self,
        example: Dict,
        architecture: Architecture,
    ) -> str:
        """Get the input portion (without answer) for generation."""
        context = example["context"][:300]
        question = example["question"]
        
        if architecture == Architecture.DECODER:
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        elif architecture == Architecture.ENCODER_DECODER:
            return f"question: {question} context: {context}"
        return f"{question} [SEP] {context}"


@register_task("squad-v2")
class SQuADv2Task(QATask):
    """
    SQuAD v2 - includes unanswerable questions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "squad_v2"

    @property
    def name(self) -> str:
        return "qa-v2"

    def get_unanswerable_examples(self, n: Optional[int] = None) -> List[Dict]:
        """Get examples without answers."""
        if self._data is None:
            self.load_data()
        
        examples = []
        for ex in self._data:
            answers = ex[self.answer_column]
            if isinstance(answers, dict):
                if not answers.get("text"):
                    examples.append({
                        "context": ex[self.context_column],
                        "question": ex[self.question_column],
                        "answer": "",
                    })
        
        if n:
            examples = examples[:n]
        return examples


@register_task("natural-questions")
class NaturalQuestionsTask(SpanExtractionTask):
    """
    Natural Questions dataset (more realistic QA).
    """

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="natural_questions",
            dataset_config="default",
            context_column="document",
            question_column="question",
            answer_column="annotations",
            **kwargs
        )

    @property
    def name(self) -> str:
        return "natural-questions"

    def _get_answer(self, example: Dict[str, Any]) -> str:
        """Extract short answer from NQ format."""
        annotations = example.get("annotations", {})
        if isinstance(annotations, dict):
            short_answers = annotations.get("short_answers", [])
            if short_answers and len(short_answers) > 0:
                # Get first short answer
                sa = short_answers[0]
                if isinstance(sa, dict):
                    start = sa.get("start_token", 0)
                    end = sa.get("end_token", 0)
                    tokens = example.get("document", {}).get("tokens", [])
                    if tokens and start < end:
                        return " ".join(tokens[start:end])
        return ""

    def load_data(
        self,
        split: str = "validation",
        max_samples: Optional[int] = 100,  # NQ is large
    ) -> None:
        """Load Natural Questions."""
        from datasets import load_dataset
        
        try:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=split,
            )
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            self._data = dataset
        except Exception as e:
            print(f"Warning: Could not load Natural Questions: {e}")
            self._data = []
