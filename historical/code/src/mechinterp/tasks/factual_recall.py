"""
Factual Recall Task (LAMA-style)

Tests whether models can recall factual knowledge.
"""

from typing import Dict, List, Optional, Any, Tuple

import torch
from mechinterp.core.base import Architecture, BaseTask
from mechinterp.core.registry import register_task


# Common factual prompts (subset of LAMA)
FACTUAL_PROMPTS = [
    # Capital cities
    {"subject": "France", "relation": "capital", "object": "Paris", 
     "template": "{subject}'s capital is [MASK].", "prompt": "The capital of {subject} is"},
    {"subject": "Germany", "relation": "capital", "object": "Berlin",
     "template": "{subject}'s capital is [MASK].", "prompt": "The capital of {subject} is"},
    {"subject": "Japan", "relation": "capital", "object": "Tokyo",
     "template": "{subject}'s capital is [MASK].", "prompt": "The capital of {subject} is"},
    {"subject": "Italy", "relation": "capital", "object": "Rome",
     "template": "{subject}'s capital is [MASK].", "prompt": "The capital of {subject} is"},
    {"subject": "Spain", "relation": "capital", "object": "Madrid",
     "template": "{subject}'s capital is [MASK].", "prompt": "The capital of {subject} is"},
    
    # Languages
    {"subject": "French", "relation": "spoken_in", "object": "France",
     "template": "{subject} is spoken in [MASK].", "prompt": "{subject} is the official language of"},
    {"subject": "German", "relation": "spoken_in", "object": "Germany",
     "template": "{subject} is spoken in [MASK].", "prompt": "{subject} is the official language of"},
    {"subject": "Japanese", "relation": "spoken_in", "object": "Japan",
     "template": "{subject} is spoken in [MASK].", "prompt": "{subject} is the official language of"},
    
    # Famous people
    {"subject": "Einstein", "relation": "field", "object": "physics",
     "template": "{subject} was famous for [MASK].", "prompt": "Albert {subject} worked in the field of"},
    {"subject": "Shakespeare", "relation": "profession", "object": "playwright",
     "template": "{subject} was a [MASK].", "prompt": "William {subject} was a famous"},
    {"subject": "Mozart", "relation": "profession", "object": "composer",
     "template": "{subject} was a [MASK].", "prompt": "Wolfgang Amadeus {subject} was a"},
    
    # Companies
    {"subject": "Apple", "relation": "ceo", "object": "Cook",
     "template": "The CEO of {subject} is [MASK].", "prompt": "The CEO of {subject} is Tim"},
    {"subject": "Microsoft", "relation": "founder", "object": "Gates",
     "template": "{subject} was founded by [MASK].", "prompt": "{subject} was founded by Bill"},
    {"subject": "Amazon", "relation": "founder", "object": "Bezos",
     "template": "{subject} was founded by [MASK].", "prompt": "{subject} was founded by Jeff"},
]


@register_task("factual_recall")
class FactualRecallTask(BaseTask):
    """
    Factual recall task using LAMA-style prompts.
    
    Tests whether models can complete factual statements.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._facts = FACTUAL_PROMPTS
        self._data = None

    @property
    def name(self) -> str:
        return "factual_recall"

    @property
    def num_classes(self) -> int:
        return -1  # Not applicable (completion task)

    def load_data(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> None:
        """Load factual prompts."""
        self._data = self._facts.copy()
        if max_samples:
            self._data = self._data[:max_samples]

    def get_facts(self) -> List[Dict[str, str]]:
        """Get all factual prompts."""
        if self._data is None:
            self.load_data()
        return self._data

    def get_positive_examples(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Get correct factual statements.
        
        Returns completed prompts with correct answers.
        """
        import random
            
        facts = self.get_facts()
        
        # Shuffle with seed for reproducible but varied sampling
        if seed is not None:
            random.seed(seed)
            facts = facts.copy()  # Don't modify original
            random.shuffle(facts)
        
        examples = [
            f"{f['prompt'].format(subject=f['subject'])} {f['object']}"
            for f in facts
        ]
        if n:
            examples = examples[:n]
        return examples

    def get_negative_examples(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Get incorrect factual statements.
        
        Returns prompts with wrong answers (shuffled objects).
        """
        import random
        if seed is not None:
            random.seed(seed)
            
        facts = self.get_facts()
        
        # Shuffle facts first for varied sampling
        if seed is not None:
            facts = facts.copy()
            random.shuffle(facts)
        
        # Shuffle objects to create incorrect pairs
        objects = [f["object"] for f in facts]
        shuffled = objects.copy()
        random.shuffle(shuffled)
        
        # Ensure no correct matches
        for i in range(len(shuffled)):
            if shuffled[i] == objects[i]:
                # Swap with next
                j = (i + 1) % len(shuffled)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        
        examples = [
            f"{f['prompt'].format(subject=f['subject'])} {obj}"
            for f, obj in zip(facts, shuffled)
        ]
        if n:
            examples = examples[:n]
        return examples

    def format_input(
        self,
        example: Dict[str, Any],
        architecture: Architecture,
    ) -> str:
        """Format factual prompt for architecture."""
        if architecture == Architecture.ENCODER:
            # Use masked template
            return example["template"].format(subject=example["subject"])
        elif architecture == Architecture.DECODER:
            # Use completion prompt
            return example["prompt"].format(subject=example["subject"])
        elif architecture == Architecture.ENCODER_DECODER:
            return f"complete: {example['prompt'].format(subject=example['subject'])}"
        return example["prompt"].format(subject=example["subject"])

    def format_for_encoder_mlm(self, fact: Dict[str, str]) -> str:
        """Format for masked language modeling."""
        return fact["template"].format(subject=fact["subject"])

    def format_for_decoder(self, fact: Dict[str, str]) -> str:
        """Format for autoregressive completion."""
        return fact["prompt"].format(subject=fact["subject"])

    def format_for_enc_dec(self, fact: Dict[str, str]) -> str:
        """Format for seq2seq."""
        prompt = fact["prompt"].format(subject=fact["subject"])
        return f"complete: {prompt}"

    def get_label(self, example: Dict[str, Any]) -> int:
        """Not applicable."""
        return -1

    def get_target(self, fact: Dict[str, str]) -> str:
        """Get the target answer for a fact."""
        return fact["object"]

    def evaluate(
        self,
        predictions: List[str],
        labels: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate factual recall.
        
        Checks if prediction contains the correct answer.
        """
        correct = 0
        exact = 0
        
        for pred, label in zip(predictions, labels):
            pred = pred.lower().strip()
            label = label.lower().strip()
            
            # Exact match
            if pred == label:
                exact += 1
                correct += 1
            # Contains match
            elif label in pred:
                correct += 1
        
        n = len(predictions)
        return {
            "accuracy": correct / n if n > 0 else 0,
            "exact_match": exact / n if n > 0 else 0,
        }

    def evaluate_model_recall(
        self,
        model,
        facts: Optional[List[Dict]] = None,
        architecture: Architecture = Architecture.DECODER,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate a model's factual recall.
        
        Args:
            model: Model wrapper
            facts: Facts to test (default: all)
            architecture: Model architecture
            top_k: Check if answer in top k predictions
            
        Returns:
            Metrics dict
        """
        if facts is None:
            facts = self.get_facts()
        
        recall_at_1 = 0
        recall_at_k = 0
        
        for fact in facts:
            target = fact["object"].lower()
            
            if architecture == Architecture.DECODER:
                prompt = self.format_for_decoder(fact)
                # Get next token probabilities
                probs = model.generate_next_token_probs(prompt)
                
                # Check top predictions
                top_tokens = torch.topk(probs[0], k=top_k)
                top_words = [
                    model.tokenizer.decode([idx]).strip().lower()
                    for idx in top_tokens.indices
                ]
                
                if target in top_words[0]:
                    recall_at_1 += 1
                if any(target in w for w in top_words):
                    recall_at_k += 1
        
        n = len(facts)
        return {
            "recall@1": recall_at_1 / n if n > 0 else 0,
            f"recall@{top_k}": recall_at_k / n if n > 0 else 0,
        }


@register_task("lama")
class LAMATask(FactualRecallTask):
    """
    Full LAMA benchmark task.
    
    Loads from the actual LAMA dataset if available.
    """

    def __init__(self, relation: str = "P36", **kwargs):
        """
        Args:
            relation: LAMA relation to test (e.g., P36 for capitals)
        """
        super().__init__(**kwargs)
        self.relation = relation

    @property
    def name(self) -> str:
        return f"lama-{self.relation}"

    def load_data(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Try to load from HuggingFace LAMA dataset.
        Falls back to built-in prompts if not available.
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset("lama", self.relation, split="train")
            
            self._data = []
            for ex in dataset:
                self._data.append({
                    "subject": ex["sub_label"],
                    "object": ex["obj_label"],
                    "template": ex["template"],
                    "prompt": ex["template"].replace("[MASK]", "").strip(),
                    "relation": self.relation,
                })
            
            if max_samples:
                self._data = self._data[:max_samples]
                
        except Exception:
            # Fall back to built-in
            super().load_data(split, max_samples)
