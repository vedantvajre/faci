"""
Analysis and visualization utilities.
"""

from mechinterp.analysis.comparison import (
    ComparisonResult,
    MethodComparison,
    aggregate_results,
    save_results,
    load_results,
)
from mechinterp.analysis.visualization import (
    plot_layer_accuracy,
    plot_circuit_heatmap,
    plot_direction_projections,
    plot_method_comparison,
    plot_token_importance,
    plot_architecture_comparison,
    create_results_report,
)

__all__ = [
    # Comparison
    "ComparisonResult",
    "MethodComparison",
    "aggregate_results",
    "save_results",
    "load_results",
    # Visualization
    "plot_layer_accuracy",
    "plot_circuit_heatmap",
    "plot_direction_projections",
    "plot_method_comparison",
    "plot_token_importance",
    "plot_architecture_comparison",
    "create_results_report",
]
