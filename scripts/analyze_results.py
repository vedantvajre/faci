#!/usr/bin/env python3
"""
Analyze aggregated multi-seed results and generate paper-ready outputs.

Usage:
    python scripts/analyze_results.py --input results/multiseed/aggregated_results.json
    
Outputs:
    - results/analysis/summary_tables.md (LaTeX-ready tables)
    - results/analysis/figures/ (visualization PNGs)
    - results/analysis/statistical_tests.json (p-values for key comparisons)
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def load_results(path: str) -> dict:
    """Load aggregated results."""
    with open(path) as f:
        return json.load(f)


def compute_effect_sizes(results: List[dict]) -> dict:
    """Compute Cohen's d effect sizes for key comparisons."""
    
    effects = {}
    
    # Group by architecture
    by_arch = defaultdict(list)
    for r in results:
        by_arch[r["architecture"]].append(r)
    
    # Decoder vs Encoder Tier 1 comparison
    decoder_t1 = []
    encoder_t1 = []
    for r in results:
        if r["architecture"] == "decoder":
            decoder_t1.extend(r["tier1_separation_values"])
        elif r["architecture"] == "encoder":
            encoder_t1.extend(r["tier1_separation_values"])
    
    if decoder_t1 and encoder_t1:
        pooled_std = np.sqrt((np.var(decoder_t1) + np.var(encoder_t1)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(encoder_t1) - np.mean(decoder_t1)) / pooled_std
            effects["encoder_vs_decoder_tier1_cohens_d"] = float(cohens_d)
    
    # Tier 1 vs Tier 2 gap for decoders
    decoder_gaps = []
    for r in results:
        if r["architecture"] == "decoder":
            for t1, t2 in zip(r["tier1_separation_values"], r["tier2_best_accuracy_values"]):
                decoder_gaps.append(t2 - t1)
    
    if decoder_gaps:
        effects["decoder_t2_minus_t1_mean"] = float(np.mean(decoder_gaps))
        effects["decoder_t2_minus_t1_std"] = float(np.std(decoder_gaps))
        # One-sample t-test: is gap significantly > 0?
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(decoder_gaps, 0)
        effects["decoder_t2_gt_t1_tstat"] = float(t_stat)
        effects["decoder_t2_gt_t1_pvalue"] = float(p_val)
    
    return effects


def generate_summary_table(results: List[dict], output_dir: str):
    """Generate summary table for paper."""
    
    # Group by (model, task)
    table_rows = []
    
    for r in sorted(results, key=lambda x: (x["architecture"], x["model"], x["task"])):
        row = {
            "Model": r["model"].replace("facebook/", ""),
            "Task": r["task"],
            "Arch": r["architecture"][:3].upper(),
            "Tier1": f"{r['tier1_separation_mean']:.3f}±{r['tier1_separation_std']:.3f}",
            "Tier2": f"{r['tier2_best_accuracy_mean']:.3f}±{r['tier2_best_accuracy_std']:.3f}",
            "Tier2_Layer": f"L{r['tier2_best_layer_mode']}",
            "Sparsity": f"{r['tier3_sparsity_mean']:.2f}±{r['tier3_sparsity_std']:.2f}",
            "Recovery": f"{r['tier3_mean_recovery_mean']:.3f}±{r['tier3_mean_recovery_std']:.3f}",
        }
        table_rows.append(row)
    
    # Generate Markdown table
    md_lines = ["# Summary Results (Mean ± Std across seeds)\n"]
    md_lines.append("| Model | Task | Arch | Tier 1 | Tier 2 (Layer) | Sparsity | Recovery |")
    md_lines.append("|-------|------|------|--------|----------------|----------|----------|")
    
    for row in table_rows:
        md_lines.append(
            f"| {row['Model']} | {row['Task']} | {row['Arch']} | "
            f"{row['Tier1']} | {row['Tier2']} ({row['Tier2_Layer']}) | "
            f"{row['Sparsity']} | {row['Recovery']} |"
        )
    
    # Add architecture summary
    md_lines.append("\n\n# Architecture Summary\n")
    
    by_arch = defaultdict(list)
    for r in results:
        by_arch[r["architecture"]].append(r)
    
    md_lines.append("| Architecture | Avg Tier 1 | Avg Tier 2 | Avg Sparsity | T1 vs T2 |")
    md_lines.append("|--------------|------------|------------|--------------|----------|")
    
    for arch in ["encoder", "decoder", "encoder_decoder"]:
        if arch not in by_arch:
            continue
        arch_results = by_arch[arch]
        
        t1_vals = [r["tier1_separation_mean"] for r in arch_results]
        t2_vals = [r["tier2_best_accuracy_mean"] for r in arch_results]
        sp_vals = [r["tier3_sparsity_mean"] for r in arch_results]
        
        t1_avg = np.mean(t1_vals)
        t2_avg = np.mean(t2_vals)
        sp_avg = np.mean(sp_vals)
        
        gap = "T2 > T1" if t2_avg > t1_avg else "T1 > T2"
        
        md_lines.append(
            f"| {arch} | {t1_avg:.3f} | {t2_avg:.3f} | {sp_avg:.2f} | {gap} |"
        )
    
    # Add task summary
    md_lines.append("\n\n# Task Summary\n")
    
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)
    
    md_lines.append("| Task | Avg Tier 1 | Avg Tier 2 | Avg Sparsity |")
    md_lines.append("|------|------------|------------|--------------|")
    
    for task in ["sentiment", "nli", "factual_recall", "qa"]:
        if task not in by_task:
            continue
        task_results = by_task[task]
        
        t1_avg = np.mean([r["tier1_separation_mean"] for r in task_results])
        t2_avg = np.mean([r["tier2_best_accuracy_mean"] for r in task_results])
        sp_avg = np.mean([r["tier3_sparsity_mean"] for r in task_results])
        
        md_lines.append(f"| {task} | {t1_avg:.3f} | {t2_avg:.3f} | {sp_avg:.2f} |")
    
    # Write to file
    output_path = os.path.join(output_dir, "summary_tables.md")
    with open(output_path, "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"Summary tables saved to: {output_path}")
    return output_path


def generate_latex_table(results: List[dict], output_dir: str):
    """Generate LaTeX table for paper."""
    
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Three-tier analysis results across architectures and tasks (mean $\pm$ std, n=5 seeds)}",
        r"\label{tab:results}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Task & Tier 1 & Tier 2 (Layer) & Sparsity & Recovery \\",
        r"\midrule",
    ]
    
    current_arch = None
    for r in sorted(results, key=lambda x: (x["architecture"], x["model"], x["task"])):
        if current_arch != r["architecture"]:
            if current_arch is not None:
                latex_lines.append(r"\midrule")
            current_arch = r["architecture"]
        
        model = r["model"].replace("facebook/", "").replace("-", " ").replace("_", " ")
        task = r["task"].replace("_", " ")
        
        latex_lines.append(
            f"{model} & {task} & "
            f"${r['tier1_separation_mean']:.2f} \\pm {r['tier1_separation_std']:.2f}$ & "
            f"${r['tier2_best_accuracy_mean']:.2f} \\pm {r['tier2_best_accuracy_std']:.2f}$ (L{r['tier2_best_layer_mode']}) & "
            f"${r['tier3_sparsity_mean']:.2f}$ & "
            f"${r['tier3_mean_recovery_mean']:.2f}$ \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    output_path = os.path.join(output_dir, "results_table.tex")
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))
    
    print(f"LaTeX table saved to: {output_path}")
    return output_path


def generate_key_findings(results: List[dict], effects: dict, output_dir: str):
    """Generate key findings summary for paper."""
    
    findings = ["# Key Findings (Statistical Validation)\n"]
    
    # Finding 1: Decoder Tier 1 vs Tier 2 dissociation
    if "decoder_t2_gt_t1_pvalue" in effects:
        p_val = effects["decoder_t2_gt_t1_pvalue"]
        mean_gap = effects["decoder_t2_minus_t1_mean"]
        findings.append(f"## Finding 1: Decoder Tier 2 > Tier 1 Dissociation")
        findings.append(f"- Mean gap (Tier 2 - Tier 1): {mean_gap:.3f} ± {effects['decoder_t2_minus_t1_std']:.3f}")
        findings.append(f"- One-sample t-test p-value: {p_val:.2e}")
        findings.append(f"- **{'Significant' if p_val < 0.05 else 'Not significant'}** at α=0.05\n")
    
    # Finding 2: Architecture ranking
    if "encoder_vs_decoder_tier1_cohens_d" in effects:
        d = effects["encoder_vs_decoder_tier1_cohens_d"]
        findings.append(f"## Finding 2: Encoder vs Decoder Tier 1")
        findings.append(f"- Cohen's d: {d:.2f} ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'} effect)")
        findings.append(f"- Encoders have {'higher' if d > 0 else 'lower'} Tier 1 separation\n")
    
    # Finding 3: Task effects
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)
    
    findings.append("## Finding 3: Task Complexity Effects")
    for task in ["sentiment", "nli", "factual_recall", "qa"]:
        if task in by_task:
            t1_vals = [r["tier1_separation_mean"] for r in by_task[task]]
            sp_vals = [r["tier3_sparsity_mean"] for r in by_task[task]]
            findings.append(f"- {task}: Tier 1 = {np.mean(t1_vals):.3f}, Sparsity = {np.mean(sp_vals):.2f}")
    
    # Finding 4: Identify high-variance results
    findings.append("\n## Finding 4: Result Stability")
    high_var = []
    for r in results:
        if r["tier1_separation_std"] > 0.05:
            high_var.append(f"- {r['model']} x {r['task']}: Tier 1 std = {r['tier1_separation_std']:.3f}")
    
    if high_var:
        findings.append("High variance results (std > 0.05):")
        findings.extend(high_var)
    else:
        findings.append("All results have low variance (std < 0.05) ✓")
    
    # Finding 5: Check for negative recovery
    findings.append("\n## Finding 5: Negative Recovery Layers")
    negative_found = []
    for r in results:
        for layer_key, stats in r.get("tier3_layer_recoveries_stats", {}).items():
            if stats["mean"] < 0:
                negative_found.append(
                    f"- {r['model']} x {r['task']} {layer_key}: {stats['mean']:.3f} ± {stats['std']:.3f}"
                )
    
    if negative_found:
        findings.append("Layers with negative mean recovery:")
        findings.extend(negative_found)
    else:
        findings.append("No layers with consistently negative recovery")
    
    output_path = os.path.join(output_dir, "key_findings.md")
    with open(output_path, "w") as f:
        f.write("\n".join(findings))
    
    print(f"Key findings saved to: {output_path}")
    return output_path


def generate_visualizations(results: List[dict], output_dir: str):
    """Generate visualization data (JSON format for plotting)."""
    
    # Prepare data for various plots
    viz_data = {
        "tier1_vs_tier2_scatter": [],
        "layer_recovery_heatmap": {},
        "sparsity_by_task": defaultdict(list),
        "architecture_comparison": defaultdict(lambda: defaultdict(list)),
    }
    
    for r in results:
        # Scatter plot data
        viz_data["tier1_vs_tier2_scatter"].append({
            "model": r["model"],
            "task": r["task"],
            "architecture": r["architecture"],
            "tier1_mean": r["tier1_separation_mean"],
            "tier1_std": r["tier1_separation_std"],
            "tier2_mean": r["tier2_best_accuracy_mean"],
            "tier2_std": r["tier2_best_accuracy_std"],
        })
        
        # Heatmap data
        key = f"{r['model']}_{r['task']}"
        viz_data["layer_recovery_heatmap"][key] = {
            "model": r["model"],
            "task": r["task"],
            "layers": {
                k: v["mean"] for k, v in r.get("tier3_layer_recoveries_stats", {}).items()
            }
        }
        
        # Sparsity by task
        viz_data["sparsity_by_task"][r["task"]].append({
            "model": r["model"],
            "sparsity": r["tier3_sparsity_mean"],
        })
        
        # Architecture comparison
        viz_data["architecture_comparison"][r["architecture"]]["tier1"].append(r["tier1_separation_mean"])
        viz_data["architecture_comparison"][r["architecture"]]["tier2"].append(r["tier2_best_accuracy_mean"])
        viz_data["architecture_comparison"][r["architecture"]]["sparsity"].append(r["tier3_sparsity_mean"])
    
    output_path = os.path.join(output_dir, "visualization_data.json")
    with open(output_path, "w") as f:
        json.dump(viz_data, f, indent=2)
    
    print(f"Visualization data saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze aggregated benchmark results")
    parser.add_argument("--input", required=True, help="Path to aggregated_results.json")
    parser.add_argument("--output-dir", default="results/analysis", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    results = data["aggregated_results"]
    
    print(f"Found {len(results)} (model, task) combinations from {data['metadata']['num_seeds']} seeds")
    
    # Generate outputs
    print("\nGenerating analysis outputs...")
    
    # Compute effect sizes
    effects = compute_effect_sizes(results)
    effects_path = os.path.join(args.output_dir, "effect_sizes.json")
    with open(effects_path, "w") as f:
        json.dump(effects, f, indent=2)
    print(f"Effect sizes saved to: {effects_path}")
    
    # Generate tables
    generate_summary_table(results, args.output_dir)
    generate_latex_table(results, args.output_dir)
    
    # Generate findings
    generate_key_findings(results, effects, args.output_dir)
    
    # Generate visualization data
    generate_visualizations(results, args.output_dir)
    
    print(f"\n✓ Analysis complete! Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
