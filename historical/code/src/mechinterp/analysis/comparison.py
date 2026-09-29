"""
Comparison utilities for evaluating methods.

Compares mechanistic interpretability (Tier 1/2/3) against baselines.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mechinterp.core.base import (
    BaseModelWrapper,
    Tier1Result,
    Tier2Result,
    Tier3Result,
)


@dataclass
class ComparisonResult:
    """Result of comparing methods."""
    model_name: str
    task_name: str
    tier1_results: Optional[Tier1Result]
    tier2_results: Optional[Tier2Result]
    tier3_results: Optional[Tier3Result]
    baseline_results: Dict[str, Dict[str, Any]]
    comparison_metrics: Dict[str, float]


class MethodComparison:
    """
    Compare mechanistic interpretability methods against baselines.
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        tier1_analyzer=None,
        tier2_analyzer=None,
        tier3_analyzer=None,
        baselines: Optional[List] = None,
    ):
        """
        Args:
            model: Model wrapper
            tier1_analyzer: Tier 1 analyzer instance
            tier2_analyzer: Tier 2 analyzer instance
            tier3_analyzer: Tier 3 analyzer instance
            baselines: List of baseline method instances
        """
        self.model = model
        self.tier1 = tier1_analyzer
        self.tier2 = tier2_analyzer
        self.tier3 = tier3_analyzer
        self.baselines = baselines or []

    def run_comparison(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        task_name: str = "unknown",
        **kwargs
    ) -> ComparisonResult:
        """
        Run all methods and compare results.
        
        Returns:
            ComparisonResult with all method outputs
        """
        # Run tier analyses
        tier1_result = None
        tier2_result = None
        tier3_result = None
        
        if self.tier1:
            tier1_result = self.tier1.analyze(positive_examples, negative_examples, **kwargs)
        
        if self.tier2:
            tier2_result = self.tier2.analyze(positive_examples, negative_examples, **kwargs)
        
        if self.tier3:
            tier3_result = self.tier3.analyze(positive_examples, negative_examples, **kwargs)
        
        # Run baselines on sample
        baseline_results = {}
        sample_texts = positive_examples[:5] + negative_examples[:5]
        
        for baseline in self.baselines:
            explanations = baseline.explain_batch(sample_texts)
            baseline_results[baseline.name] = {
                "explanations": explanations,
                "is_causal": baseline.is_causal,
            }
        
        # Compute comparison metrics
        comparison_metrics = self._compute_comparison_metrics(
            tier1_result, tier2_result, tier3_result, baseline_results
        )
        
        return ComparisonResult(
            model_name=self.model.model_name,
            task_name=task_name,
            tier1_results=tier1_result,
            tier2_results=tier2_result,
            tier3_results=tier3_result,
            baseline_results=baseline_results,
            comparison_metrics=comparison_metrics,
        )

    def _compute_comparison_metrics(
        self,
        tier1: Optional[Tier1Result],
        tier2: Optional[Tier2Result],
        tier3: Optional[Tier3Result],
        baselines: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Compute metrics comparing methods."""
        metrics = {}
        
        # Tier 1 metrics
        if tier1:
            metrics["tier1_separation"] = tier1.separation
        
        # Tier 2 metrics
        if tier2:
            metrics["tier2_best_accuracy"] = tier2.best_accuracy
            metrics["tier2_emergence_layer"] = tier2.emergence_layer
            metrics["tier2_best_layer"] = tier2.best_layer
        
        # Tier 3 metrics
        if tier3:
            metrics["tier3_circuit_size"] = tier3.circuit_size
            metrics["tier3_sparsity"] = tier3.sparsity
        
        # Baseline metrics
        for name, data in baselines.items():
            metrics[f"baseline_{name}_is_causal"] = 1.0 if data["is_causal"] else 0.0
        
        return metrics

    def compare_token_importance(
        self,
        text: str,
        target: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compare token importance scores across methods.
        
        Returns DataFrame with tokens and scores from each method.
        """
        results = {"token": []}
        
        # Get baseline explanations
        for baseline in self.baselines:
            explanation = baseline.explain(text, target)
            
            if not results["token"]:
                results["token"] = explanation["tokens"]
            
            results[baseline.name] = explanation["importance_scores"].tolist()
        
        return pd.DataFrame(results)

    def evaluate_causal_faithfulness(
        self,
        text: str,
        important_tokens: List[int],
        target: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate if methods correctly identify known-important tokens.
        
        Args:
            text: Input text
            important_tokens: Indices of actually important tokens
            target: Target class
            
        Returns:
            Dict mapping method name to faithfulness score
        """
        faithfulness = {}
        
        for baseline in self.baselines:
            explanation = baseline.explain(text, target)
            scores = explanation["importance_scores"]
            
            # Get top-k predicted important tokens
            k = len(important_tokens)
            if len(scores) > 0:
                predicted_important = set(np.argsort(scores)[-k:])
                actual_important = set(important_tokens)
                
                # Precision
                overlap = len(predicted_important & actual_important)
                faithfulness[baseline.name] = overlap / k if k > 0 else 0.0
            else:
                faithfulness[baseline.name] = 0.0
        
        return faithfulness


def aggregate_results(
    results: List[ComparisonResult],
) -> pd.DataFrame:
    """
    Aggregate comparison results into a summary DataFrame.
    
    Args:
        results: List of ComparisonResult objects
        
    Returns:
        DataFrame with summary statistics
    """
    rows = []
    
    for result in results:
        row = {
            "model": result.model_name,
            "task": result.task_name,
        }
        row.update(result.comparison_metrics)
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_results(
    results: List[ComparisonResult],
    output_dir: str,
) -> None:
    """Save comparison results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = aggregate_results(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    
    # Save individual results
    for result in results:
        filename = f"{result.model_name}_{result.task_name}.json"
        filepath = output_dir / filename
        
        # Convert to serializable format
        data = {
            "model_name": result.model_name,
            "task_name": result.task_name,
            "comparison_metrics": result.comparison_metrics,
        }
        
        if result.tier1_results:
            data["tier1"] = {
                "separation": result.tier1_results.separation,
                "layer": result.tier1_results.layer,
            }
        
        if result.tier2_results:
            data["tier2"] = {
                "best_accuracy": result.tier2_results.best_accuracy,
                "best_layer": result.tier2_results.best_layer,
                "emergence_layer": result.tier2_results.emergence_layer,
                "layer_accuracies": result.tier2_results.layer_accuracies,
            }
        
        if result.tier3_results:
            data["tier3"] = {
                "circuit_size": result.tier3_results.circuit_size,
                "sparsity": result.tier3_results.sparsity,
                "total_components": result.tier3_results.total_components,
            }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


def load_results(
    output_dir: str,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Load comparison results from disk."""
    output_dir = Path(output_dir)
    
    summary = pd.read_csv(output_dir / "summary.csv")
    
    individual = []
    for filepath in output_dir.glob("*.json"):
        with open(filepath) as f:
            individual.append(json.load(f))
    
    return summary, individual
