"""
Tier 2: Linear Probing Analysis

Trains and evaluates linear probes to track information across layers.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from mechinterp.core.base import (
    BaseModelWrapper,
    BaseTierAnalyzer,
    ExtractionPosition,
    Tier2Result,
)
from mechinterp.core.registry import register_tier


@register_tier("tier2")
class Tier2Analyzer(BaseTierAnalyzer):
    """
    Tier 2: Linear Probing Analysis
    
    Tests whether information is linearly accessible at each layer.
    
    Methods:
    - train_probe: Train a linear probe at a specific layer
    - probe_all_layers: Train probes at all layers
    - find_emergence_layer: Find where information first emerges
    - compute_transfer: Test if probe transfers to new distribution
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        position: Optional[ExtractionPosition] = None,
        **kwargs
    ):
        """
        Initialize Tier 2 analyzer.
        
        Args:
            model: Model wrapper
            position: Extraction position (uses model default if None)
        """
        super().__init__(model, **kwargs)
        self.position = position or model.default_position
        self._probes: Dict[int, LogisticRegression] = {}

    @property
    def tier(self) -> int:
        return 2

    def _get_activations(
        self,
        texts: List[str],
        layer: int,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract activations for a list of texts at specified layer."""
        all_activations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            activations = self.model.get_embedding(
                batch,
                layer=layer,
                position=self.position,
            )
            all_activations.append(activations.float().cpu().numpy())
        
        return np.vstack(all_activations)

    def train_probe(
        self,
        texts: List[str],
        labels: List[int],
        layer: int,
        test_size: float = 0.2,
        batch_size: int = 32,
        random_state: int = 42,
        **sklearn_kwargs
    ) -> Tuple[LogisticRegression, Dict[str, float]]:
        """
        Train a linear probe at a specific layer.
        
        Args:
            texts: Input texts
            labels: Binary labels (0 or 1)
            layer: Layer to probe
            test_size: Fraction for test set
            batch_size: Batch size for extraction
            random_state: Random seed
            **sklearn_kwargs: Additional args for LogisticRegression
            
        Returns:
            (trained probe, metrics dict)
        """
        # Get activations
        X = self._get_activations(texts, layer, batch_size)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Filter out non-sklearn kwargs
        valid_sklearn_args = {
            'penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling',
            'class_weight', 'solver', 'max_iter', 'multi_class', 'verbose',
            'warm_start', 'n_jobs', 'l1_ratio', 'random_state'
        }
        filtered_kwargs = {k: v for k, v in sklearn_kwargs.items() if k in valid_sklearn_args}
        
        # Train probe
        defaults = {"max_iter": 1000, "random_state": random_state}
        defaults.update(filtered_kwargs)
        probe = LogisticRegression(**defaults)
        probe.fit(X_train, y_train)
        
        # Evaluate
        y_pred = probe.predict(X_test)
        y_prob = probe.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "train_accuracy": probe.score(X_train, y_train),
        }
        
        # Store probe
        self._probes[layer] = probe
        
        return probe, metrics

    def probe_all_layers(
        self,
        texts: List[str],
        labels: List[int],
        layers: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[int, Dict[str, float]]:
        """
        Train probes at all layers.
        
        Args:
            texts: Input texts
            labels: Binary labels
            layers: Which layers to probe (default: all)
            
        Returns:
            Dictionary mapping layer to metrics
        """
        if layers is None:
            layers = list(range(self.model.num_layers + 1))
        
        results = {}
        for layer in tqdm(layers, desc="Probing layers"):
            _, metrics = self.train_probe(texts, labels, layer, **kwargs)
            results[layer] = metrics
        
        return results

    def find_emergence_layer(
        self,
        layer_results: Dict[int, Dict[str, float]],
        threshold: float = 0.6,
        metric: str = "accuracy",
    ) -> int:
        """
        Find the first layer where performance exceeds threshold.
        
        Args:
            layer_results: Results from probe_all_layers
            threshold: Performance threshold
            metric: Which metric to use
            
        Returns:
            First layer index exceeding threshold (-1 if none)
        """
        for layer in sorted(layer_results.keys()):
            if layer_results[layer][metric] >= threshold:
                return layer
        return -1

    def get_best_layer(
        self,
        layer_results: Dict[int, Dict[str, float]],
        metric: str = "accuracy",
    ) -> Tuple[int, float]:
        """Find the layer with best performance."""
        best_layer = max(layer_results.keys(), key=lambda l: layer_results[l][metric])
        return best_layer, layer_results[best_layer][metric]

    def compute_transfer(
        self,
        train_texts: List[str],
        train_labels: List[int],
        test_texts: List[str],
        test_labels: List[int],
        layer: int,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Test if probe transfers to a new distribution.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            test_texts: Test texts (different distribution)
            test_labels: Test labels
            layer: Layer to probe
            
        Returns:
            Metrics on the transfer set
        """
        # Get activations
        X_train = self._get_activations(train_texts, layer, batch_size)
        y_train = np.array(train_labels)
        X_test = self._get_activations(test_texts, layer, batch_size)
        y_test = np.array(test_labels)
        
        # Train probe on training distribution
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_train, y_train)
        
        # Evaluate on test distribution
        y_pred = probe.predict(X_test)
        y_prob = probe.predict_proba(X_test)[:, 1]
        
        return {
            "transfer_accuracy": accuracy_score(y_test, y_pred),
            "transfer_auc": roc_auc_score(y_test, y_prob),
            "train_accuracy": probe.score(X_train, y_train),
        }

    def compute_transfer_matrix(
        self,
        data_by_split: Dict[str, Tuple[List[str], List[int]]],
        layer: int,
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute transfer matrix between multiple distributions.
        
        Args:
            data_by_split: Dict mapping split name to (texts, labels)
            layer: Layer to probe
            
        Returns:
            Nested dict: results[train_split][test_split] = accuracy
        """
        results = {}
        split_names = list(data_by_split.keys())
        
        for train_split in split_names:
            results[train_split] = {}
            train_texts, train_labels = data_by_split[train_split]
            
            # Get train activations and fit probe
            X_train = self._get_activations(train_texts, layer, batch_size)
            y_train = np.array(train_labels)
            
            probe = LogisticRegression(max_iter=1000)
            probe.fit(X_train, y_train)
            
            # Test on all splits
            for test_split in split_names:
                test_texts, test_labels = data_by_split[test_split]
                X_test = self._get_activations(test_texts, layer, batch_size)
                y_test = np.array(test_labels)
                
                accuracy = probe.score(X_test, y_test)
                results[train_split][test_split] = accuracy
        
        return results

    def analyze(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        **kwargs
    ) -> Tier2Result:
        """
        Run full Tier 2 analysis.
        
        Returns:
            Tier2Result with layer accuracies and emergence info
        """
        # Combine examples
        texts = positive_examples + negative_examples
        labels = [1] * len(positive_examples) + [0] * len(negative_examples)
        
        # Probe all layers
        layer_results = self.probe_all_layers(texts, labels, **kwargs)
        
        # Extract accuracy per layer
        layer_accuracies = {l: r["accuracy"] for l, r in layer_results.items()}
        
        # Find emergence and best
        emergence = self.find_emergence_layer(layer_results, threshold=kwargs.get("threshold", 0.6))
        best_layer, best_acc = self.get_best_layer(layer_results)
        
        return Tier2Result(
            layer_accuracies=layer_accuracies,
            emergence_layer=emergence,
            best_layer=best_layer,
            best_accuracy=best_acc,
            probes=self._probes if kwargs.get("store_probes", False) else {},
        )
