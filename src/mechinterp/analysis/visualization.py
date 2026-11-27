"""
Visualization utilities for mechanistic interpretability results.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

# Lazy imports for optional dependencies
def _get_matplotlib():
    import matplotlib.pyplot as plt
    return plt

def _get_seaborn():
    import seaborn as sns
    return sns


def plot_layer_accuracy(
    layer_accuracies: Dict[int, float],
    title: str = "Probe Accuracy by Layer",
    save_path: Optional[str] = None,
    emergence_threshold: float = 0.6,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot probe accuracy across layers (Tier 2 results).
    
    Args:
        layer_accuracies: Dict mapping layer index to accuracy
        title: Plot title
        save_path: Path to save figure
        emergence_threshold: Threshold line to show
        figsize: Figure size
    """
    plt = _get_matplotlib()
    
    layers = sorted(layer_accuracies.keys())
    accuracies = [layer_accuracies[l] for l in layers]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(layers, accuracies, marker='o', linewidth=2, markersize=8)
    ax.axhline(y=emergence_threshold, color='r', linestyle='--', 
               label=f'Emergence threshold ({emergence_threshold})')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Probe Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark emergence layer
    for i, (l, acc) in enumerate(zip(layers, accuracies)):
        if acc >= emergence_threshold:
            ax.axvline(x=l, color='g', linestyle=':', alpha=0.5)
            ax.annotate(f'Emergence (L{l})', xy=(l, acc), 
                       xytext=(l+0.5, acc-0.1),
                       fontsize=10, color='g')
            break
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_circuit_heatmap(
    component_effects: Dict[str, float],
    num_layers: int,
    num_heads: int,
    title: str = "Circuit Component Effects",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot heatmap of attention head effects (Tier 3 results).
    
    Args:
        component_effects: Dict mapping component name to effect
        num_layers: Number of layers
        num_heads: Number of attention heads
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt = _get_matplotlib()
    sns = _get_seaborn()
    
    # Create effect matrix
    effects = np.zeros((num_layers, num_heads))
    
    for name, effect in component_effects.items():
        # Parse component name (e.g., "layer_5_head_3" or "layer_5")
        parts = name.split('_')
        if 'head' in parts:
            layer_idx = int(parts[parts.index('layer') + 1])
            head_idx = int(parts[parts.index('head') + 1])
            if layer_idx < num_layers and head_idx < num_heads:
                effects[layer_idx, head_idx] = effect
        elif 'layer' in parts:
            layer_idx = int(parts[parts.index('layer') + 1])
            if layer_idx < num_layers:
                effects[layer_idx, :] = effect
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        effects,
        ax=ax,
        cmap='RdYlBu_r',
        center=0,
        xticklabels=[f'H{i}' for i in range(num_heads)],
        yticklabels=[f'L{i}' for i in range(num_layers)],
        cbar_kws={'label': 'Effect'},
    )
    
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_direction_projections(
    positive_projections: List[float],
    negative_projections: List[float],
    title: str = "Feature Direction Projections",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot histogram of projections onto feature direction (Tier 1 results).
    
    Args:
        positive_projections: Projections of positive examples
        negative_projections: Projections of negative examples
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt = _get_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(positive_projections, bins=30, alpha=0.7, label='Positive', color='green')
    ax.hist(negative_projections, bins=30, alpha=0.7, label='Negative', color='red')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Projection onto Direction', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_method_comparison(
    comparison_data: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot comparison of methods across models/tasks.
    
    Args:
        comparison_data: Nested dict {method: {model_task: score}}
        metric: Metric being compared
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt = _get_matplotlib()
    
    methods = list(comparison_data.keys())
    model_tasks = list(comparison_data[methods[0]].keys())
    
    x = np.arange(len(model_tasks))
    width = 0.8 / len(methods)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, method in enumerate(methods):
        scores = [comparison_data[method].get(mt, 0) for mt in model_tasks]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=method)
    
    ax.set_xlabel('Model / Task', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_token_importance(
    tokens: List[str],
    importance_scores: Dict[str, List[float]],
    title: str = "Token Importance Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """
    Plot token importance from multiple methods.
    
    Args:
        tokens: List of tokens
        importance_scores: Dict mapping method name to importance scores
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt = _get_matplotlib()
    
    methods = list(importance_scores.keys())
    n_tokens = len(tokens)
    n_methods = len(methods)
    
    fig, axes = plt.subplots(n_methods, 1, figsize=figsize, sharex=True)
    if n_methods == 1:
        axes = [axes]
    
    for ax, method in zip(axes, methods):
        scores = importance_scores[method]
        
        # Normalize scores
        scores = np.array(scores)
        if scores.max() > 0:
            scores = scores / scores.max()
        
        colors = plt.cm.RdYlBu_r(scores)
        ax.bar(range(n_tokens), scores, color=colors)
        ax.set_ylabel(method, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    axes[-1].set_xticks(range(n_tokens))
    axes[-1].set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    axes[-1].set_xlabel('Token', fontsize=12)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_architecture_comparison(
    results_by_arch: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    title: str = "Comparison Across Architectures",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot comparison across different architectures.
    
    Args:
        results_by_arch: {architecture: {task: metric_value}}
        metric: Metric being compared
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt = _get_matplotlib()
    sns = _get_seaborn()
    
    # Convert to DataFrame format
    data = []
    for arch, tasks in results_by_arch.items():
        for task, value in tasks.items():
            data.append({
                "Architecture": arch,
                "Task": task,
                metric: value,
            })
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(data=df, x="Task", y=metric, hue="Architecture", ax=ax)
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title='Architecture')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_results_report(
    results: Dict[str, Any],
    output_dir: str,
    format: str = "html",
) -> str:
    """
    Create a comprehensive results report.
    
    Args:
        results: Dict with all results
        output_dir: Directory to save report
        format: Report format ('html' or 'markdown')
        
    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "html":
        report_path = output_dir / "report.html"
        content = _generate_html_report(results)
    else:
        report_path = output_dir / "report.md"
        content = _generate_markdown_report(results)
    
    with open(report_path, "w") as f:
        f.write(content)
    
    return str(report_path)


def _generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate markdown report."""
    lines = [
        "# Mechanistic Interpretability Results Report",
        "",
        "## Summary",
        "",
    ]
    
    if "summary" in results:
        lines.append("| Model | Task | Tier 1 | Tier 2 | Tier 3 |")
        lines.append("|-------|------|--------|--------|--------|")
        for row in results["summary"]:
            lines.append(
                f"| {row.get('model', '')} | {row.get('task', '')} | "
                f"{row.get('tier1_separation', 'N/A'):.3f} | "
                f"{row.get('tier2_best_accuracy', 'N/A'):.3f} | "
                f"{row.get('tier3_sparsity', 'N/A'):.3f} |"
            )
    
    lines.extend([
        "",
        "## Detailed Results",
        "",
    ])
    
    return "\n".join(lines)


def _generate_html_report(results: Dict[str, Any]) -> str:
    """Generate HTML report."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mechanistic Interpretability Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            h1, h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>Mechanistic Interpretability Results Report</h1>
        <h2>Summary</h2>
        <p>Results from three-tier mechanistic interpretability analysis.</p>
    </body>
    </html>
    """
    return html
