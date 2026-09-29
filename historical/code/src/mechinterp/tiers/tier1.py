"""
Tier 1: Feature Direction Analysis

Computes and analyzes feature directions in activation space.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mechinterp.core.base import (
    BaseModelWrapper,
    BaseTierAnalyzer,
    ExtractionPosition,
    Tier1Result,
)
from mechinterp.core.registry import register_tier
from mechinterp.core.utils import compute_direction, project_onto_direction, cosine_similarity


@register_tier("tier1")
class Tier1Analyzer(BaseTierAnalyzer):
    """
    Tier 1: Feature Direction Analysis
    
    Tests whether a model represents a concept as a direction in activation space.
    
    Methods:
    - compute_direction: Find the direction separating positive/negative examples
    - project: Score new examples by projection onto direction
    - measure_separation: Quantify how well direction separates classes
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        layer: int = -1,
        position: Optional[ExtractionPosition] = None,
        **kwargs
    ):
        """
        Initialize Tier 1 analyzer.
        
        Args:
            model: Model wrapper
            layer: Layer to extract from (-1 for final)
            position: Extraction position (uses model default if None)
        """
        super().__init__(model, **kwargs)
        self.layer = layer
        self.position = position or model.default_position
        self._directions: Dict[str, Tensor] = {}

    @property
    def tier(self) -> int:
        return 1

    def _get_activations(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> Tensor:
        """Extract activations for a list of texts."""
        all_activations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            activations = self.model.get_embedding(
                batch,
                layer=self.layer,
                position=self.position,
            )
            all_activations.append(activations.cpu())
        
        return torch.cat(all_activations, dim=0)

    def compute_feature_direction(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        name: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> Tensor:
        """
        Compute feature direction from positive and negative examples.
        
        Args:
            positive_texts: Examples with the feature
            negative_texts: Examples without the feature
            name: Optional name to store direction for later use
            normalize: Whether to L2 normalize the direction
            batch_size: Batch size for processing
            
        Returns:
            Direction vector of shape (hidden_size,)
        """
        # Get activations
        pos_activations = self._get_activations(positive_texts, batch_size)
        neg_activations = self._get_activations(negative_texts, batch_size)
        
        # Compute direction
        direction = compute_direction(pos_activations, neg_activations, normalize)
        
        # Store if named
        if name:
            self._directions[name] = direction
        
        return direction

    def project(
        self,
        texts: Union[str, List[str]],
        direction: Optional[Tensor] = None,
        direction_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> Tensor:
        """
        Project texts onto a feature direction.
        
        Args:
            texts: Input text(s)
            direction: Direction vector (or use direction_name)
            direction_name: Name of stored direction
            batch_size: Batch size for processing
            
        Returns:
            Projection scores
        """
        if direction is None:
            if direction_name is None:
                raise ValueError("Must provide direction or direction_name")
            direction = self._directions[direction_name]
        
        if isinstance(texts, str):
            texts = [texts]
        
        activations = self._get_activations(texts, batch_size)
        return project_onto_direction(activations, direction)

    def measure_separation(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        direction: Optional[Tensor] = None,
        direction_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Measure how well direction separates positive from negative examples.
        
        Returns:
            Dictionary with separation metrics (AUC, accuracy, mean difference)
        """
        if direction is None:
            if direction_name is None:
                raise ValueError("Must provide direction or direction_name")
            direction = self._directions[direction_name]
        
        # Get projections
        pos_proj = self.project(positive_texts, direction, batch_size=batch_size)
        neg_proj = self.project(negative_texts, direction, batch_size=batch_size)
        
        # Compute metrics
        all_proj = torch.cat([pos_proj, neg_proj]).numpy()
        labels = np.array([1] * len(pos_proj) + [0] * len(neg_proj))
        
        auc = roc_auc_score(labels, all_proj)
        
        # Simple threshold accuracy (at 0)
        preds = (all_proj > 0).astype(int)
        accuracy = (preds == labels).mean()
        
        # Mean difference
        mean_diff = pos_proj.mean().item() - neg_proj.mean().item()
        
        return {
            "auc": auc,
            "accuracy": accuracy,
            "mean_difference": mean_diff,
            "positive_mean": pos_proj.mean().item(),
            "negative_mean": neg_proj.mean().item(),
        }

    def compare_directions(
        self,
        direction1: Tensor,
        direction2: Tensor,
    ) -> float:
        """Compute cosine similarity between two directions."""
        return cosine_similarity(direction1, direction2)

    def analyze(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        **kwargs
    ) -> Tier1Result:
        """
        Run full Tier 1 analysis.
        
        Returns:
            Tier1Result with direction and metrics
        """
        batch_size = kwargs.get("batch_size", 32)
        
        # Compute direction
        direction = self.compute_feature_direction(
            positive_examples,
            negative_examples,
            batch_size=batch_size,
        )
        
        # Get projections
        pos_proj = self.project(positive_examples, direction, batch_size=batch_size)
        neg_proj = self.project(negative_examples, direction, batch_size=batch_size)
        
        # Compute separation
        metrics = self.measure_separation(
            positive_examples,
            negative_examples,
            direction,
            batch_size=batch_size,
        )
        
        return Tier1Result(
            direction=direction,
            layer=self.layer,
            positive_projections=pos_proj.tolist(),
            negative_projections=neg_proj.tolist(),
            separation=metrics["auc"],
            position=self.position,
        )

    def analyze_across_layers(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        layers: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[int, Tier1Result]:
        """
        Run Tier 1 analysis across multiple layers.
        
        Args:
            positive_examples: Examples with feature
            negative_examples: Examples without feature
            layers: Which layers to analyze (default: all)
            
        Returns:
            Dictionary mapping layer index to Tier1Result
        """
        if layers is None:
            layers = list(range(self.model.num_layers + 1))  # Include embedding layer
        
        results = {}
        original_layer = self.layer
        
        for layer in tqdm(layers, desc="Tier 1 across layers"):
            self.layer = layer
            results[layer] = self.analyze(positive_examples, negative_examples, **kwargs)
        
        self.layer = original_layer
        return results
