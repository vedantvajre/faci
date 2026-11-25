#!/usr/bin/env python3
"""
Generate visualizations for paper from aggregated results.

Usage:
    python scripts/plot_results.py --input results/multiseed/aggregated_results.json
    
Outputs PNG figures to results/figures/
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

ARCH_COLORS = {
    "encoder": "#2ecc71",       # Green
    "decoder": "#e74c3c",       # Red  
    "encoder_decoder": "#3498db" # Blue
}

ARCH_MARKERS = {
    "encoder": "o",
    "decoder": "s",
    "encoder_decoder": "^"
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_tier1_vs_tier2(results: list, output_dir: str):
    """Scatter plot: Tier 1 vs Tier 2 by architecture."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for r in results:
        arch = r["architecture"]
        ax.errorbar(
            r["tier1_separation_mean"],
            r["tier2_best_accuracy_mean"],
            xerr=r["tier1_separation_std"],
            yerr=r["tier2_best_accuracy_std"],
            fmt=ARCH_MARKERS[arch],
            color=ARCH_COLORS[arch],
            markersize=10,
            capsize=3,
            alpha=0.7,
            label=arch if arch not in [l.get_label() for l in ax.lines] else ""
        )
    
    # Add diagonal line (Tier 1 = Tier 2)
    lims = [0.4, 1.0]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Tier 1 = Tier 2')
    
    ax.set_xlabel('Tier 1 Separation', fontsize=12)
    ax.set_ylabel('Tier 2 Best Accuracy', fontsize=12)
    ax.set_title('Tier 1 vs Tier 2: Architecture Comparison', fontsize=14)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "tier1_vs_tier2.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_recovery_heatmap(results: list, output_dir: str, task: str = "sentiment"):
    """Heatmap of layer recovery by model."""
    
    # Filter to specific task
    task_results = [r for r in results if r["task"] == task]
    
    if not task_results:
        print(f"No results for task: {task}")
        return
    
    # Build matrix
    models = []
    max_layers = 0
    
    for r in task_results:
        models.append(r["model"].replace("facebook/", ""))
        n_layers = len(r.get("tier3_layer_recoveries_stats", {}))
        max_layers = max(max_layers, n_layers)
    
    matrix = np.full((len(models), max_layers), np.nan)
    
    for i, r in enumerate(task_results):
        stats = r.get("tier3_layer_recoveries_stats", {})
        for j in range(len(stats)):
            key = f"layer_{j}"
            if key in stats:
                matrix[i, j] = stats[key]["mean"]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Custom colormap: red for low, white for 1.0, blue for high
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        center=1.0,
        vmin=0,
        vmax=1.5,
        xticklabels=[f"L{i}" for i in range(max_layers)],
        yticklabels=models,
        cbar_kws={"label": "Recovery"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8}
    )
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(f'Layer Recovery Heatmap ({task.upper()})', fontsize=14)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"layer_recovery_{task}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_architecture_comparison(results: list, output_dir: str):
    """Bar chart comparing architectures across metrics."""
    
    # Aggregate by architecture
    by_arch = defaultdict(lambda: {"tier1": [], "tier2": [], "sparsity": []})
    
    for r in results:
        arch = r["architecture"]
        by_arch[arch]["tier1"].append(r["tier1_separation_mean"])
        by_arch[arch]["tier2"].append(r["tier2_best_accuracy_mean"])
        by_arch[arch]["sparsity"].append(r["tier3_sparsity_mean"])
    
    # Compute means and stds
    archs = ["encoder", "decoder", "encoder_decoder"]
    metrics = ["tier1", "tier2", "sparsity"]
    metric_labels = ["Tier 1 Separation", "Tier 2 Accuracy", "Circuit Sparsity"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(archs))
    width = 0.6
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        means = [np.mean(by_arch[a][metric]) for a in archs]
        stds = [np.std(by_arch[a][metric]) for a in archs]
        colors = [ARCH_COLORS[a] for a in archs]
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, color=colors, alpha=0.8)
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["Encoder", "Decoder", "Enc-Dec"], fontsize=10)
        ax.set_title(label, fontsize=12)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.02,
                f'{mean:.2f}',
                ha='center', va='bottom', fontsize=9
            )
    
    plt.suptitle('Architecture Comparison (Averaged Across Tasks)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "architecture_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_task_comparison(results: list, output_dir: str):
    """Bar chart comparing tasks across metrics."""
    
    # Aggregate by task
    by_task = defaultdict(lambda: {"tier1": [], "tier2": [], "sparsity": []})
    
    for r in results:
        task = r["task"]
        by_task[task]["tier1"].append(r["tier1_separation_mean"])
        by_task[task]["tier2"].append(r["tier2_best_accuracy_mean"])
        by_task[task]["sparsity"].append(r["tier3_sparsity_mean"])
    
    tasks = list(by_task.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(tasks))
    width = 0.6
    
    metrics = ["tier1", "tier2", "sparsity"]
    metric_labels = ["Tier 1 Separation", "Tier 2 Accuracy", "Circuit Sparsity"]
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        means = [np.mean(by_task[t][metric]) for t in tasks]
        stds = [np.std(by_task[t][metric]) for t in tasks]
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, alpha=0.8)
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=9)
        ax.set_title(label, fontsize=12)
        
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f'{mean:.2f}',
                ha='center', va='bottom', fontsize=9
            )
    
    plt.suptitle('Task Comparison (Averaged Across Models)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "task_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_tier1_tier2_gap(results: list, output_dir: str):
    """Show Tier 2 - Tier 1 gap by architecture (key finding visualization)."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute gaps
    data = []
    for r in results:
        gap = r["tier2_best_accuracy_mean"] - r["tier1_separation_mean"]
        data.append({
            "model": r["model"].replace("facebook/", ""),
            "task": r["task"],
            "architecture": r["architecture"],
            "gap": gap
        })
    
    # Group by architecture
    by_arch = defaultdict(list)
    for d in data:
        by_arch[d["architecture"]].append(d["gap"])
    
    archs = ["encoder", "decoder", "encoder_decoder"]
    positions = range(len(archs))
    
    # Box plot
    bp_data = [by_arch[a] for a in archs]
    bp = ax.boxplot(bp_data, positions=positions, widths=0.5, patch_artist=True)
    
    # Color boxes
    for patch, arch in zip(bp['boxes'], archs):
        patch.set_facecolor(ARCH_COLORS[arch])
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, arch in enumerate(archs):
        y = by_arch[arch]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, color='black', s=30, zorder=3)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No gap')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(["Encoder", "Decoder", "Enc-Dec"], fontsize=11)
    ax.set_ylabel("Tier 2 - Tier 1 Gap", fontsize=12)
    ax.set_title("Tier 1 vs Tier 2 Dissociation by Architecture", fontsize=14)
    
    # Add annotation
    ax.annotate(
        "Decoder: Tier 2 > Tier 1\n(information accessible\nbut non-linear)",
        xy=(1, np.mean(by_arch["decoder"])),
        xytext=(1.5, 0.2),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "tier1_tier2_gap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper visualizations")
    parser.add_argument("--input", required=True, help="Path to aggregated_results.json")
    parser.add_argument("--output-dir", default="results/figures", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    results = data["aggregated_results"]
    
    print(f"Generating visualizations for {len(results)} (model, task) combinations...")
    
    # Generate all plots
    plot_tier1_vs_tier2(results, args.output_dir)
    plot_architecture_comparison(results, args.output_dir)
    plot_task_comparison(results, args.output_dir)
    plot_tier1_tier2_gap(results, args.output_dir)
    
    # Generate heatmaps for each task
    tasks = list(set(r["task"] for r in results))
    for task in tasks:
        plot_layer_recovery_heatmap(results, args.output_dir, task)
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
