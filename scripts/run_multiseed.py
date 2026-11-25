#!/usr/bin/env python3
"""
Multi-seed benchmark runner with all fixes.

Copy this file to: scripts/run_multiseed.py

Usage:
    python scripts/run_multiseed.py --config configs/experiments/benchmark.yaml --seeds 42 123 456 789 1000
    python scripts/run_multiseed.py --config configs/experiments/benchmark.yaml  # Uses default seeds
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

import yaml


DEFAULT_SEEDS = [42, 123, 456, 789, 1000]


def run_benchmark_with_seed(config_path: str, seed: int, output_dir: str) -> str:
    """Run benchmark with a specific seed and return results path."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create seed-specific output directory
    seed_output_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)
    
    # Create temporary config with this seed and output_dir
    config["seed"] = seed
    config["output_dir"] = seed_output_dir  # CRITICAL: Set output_dir
    
    temp_config_path = os.path.join(seed_output_dir, "config.yaml")
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)
    
    # Run benchmark
    print(f"\n{'='*60}")
    print(f"Running benchmark with seed={seed}")
    print(f"Output: {seed_output_dir}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        "scripts/run_benchmark.py",
        "--config", temp_config_path
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"WARNING: Benchmark with seed={seed} may have had errors (return code: {result.returncode})")
    
    # Return path to results
    results_path = os.path.join(seed_output_dir, "benchmark_results.json")
    return results_path


def safe_mean(values):
    """Compute mean, filtering NaN and None."""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def safe_std(values):
    """Compute std, filtering NaN and None."""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.std(valid)) if valid else 0.0


def safe_median(values):
    """Compute median, filtering NaN and None."""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.median(valid)) if valid else 0.0


def aggregate_results(result_paths: List[str], output_dir: str) -> Dict[str, Any]:
    """Aggregate results from multiple seeds."""
    
    all_results = []
    seeds_used = []
    
    print(f"\nAggregating results from {len(result_paths)} paths...")
    
    for path in result_paths:
        print(f"  Checking: {path}")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                all_results.append(data)
                seeds_used.append(data["metadata"].get("seed", "unknown"))
                print(f"    ✓ Loaded (seed={seeds_used[-1]})")
            except json.JSONDecodeError as e:
                print(f"    ✗ Invalid JSON: {e}")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        else:
            print(f"    ✗ File not found")
    
    if not all_results:
        # Try to find results in alternate locations
        print("\nSearching for results in alternate locations...")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file == "benchmark_results.json":
                    path = os.path.join(root, file)
                    print(f"  Found: {path}")
                    try:
                        with open(path) as f:
                            data = json.load(f)
                        all_results.append(data)
                        seeds_used.append(data["metadata"].get("seed", "unknown"))
                    except:
                        pass
    
    if not all_results:
        raise ValueError(
            f"No results found to aggregate!\n"
            f"Checked paths: {result_paths}\n"
            f"Output dir: {output_dir}"
        )
    
    print(f"\nFound {len(all_results)} result files to aggregate")
    
    # Build aggregated structure
    # Group by model-task combination
    grouped = {}
    
    for result_set in all_results:
        for result in result_set.get("results", []):
            key = (result["model"], result["task"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
    
    # Aggregate each group
    aggregated_results = []
    
    for (model, task), results in grouped.items():
        agg = {
            "model": model,
            "task": task,
            "architecture": results[0].get("architecture", "unknown"),
            "num_layers": results[0].get("num_layers", 12),
            "num_seeds": len(results),
        }
        
        # Aggregate Tier 1
        t1_values = [r.get("tier1_separation", 0) for r in results if "tier1_separation" in r]
        if t1_values:
            agg["tier1_separation_mean"] = safe_mean(t1_values)
            agg["tier1_separation_std"] = safe_std(t1_values)
            agg["tier1_separation_values"] = t1_values
        
        # Aggregate Tier 2
        t2_values = [r.get("tier2_best_accuracy", 0) for r in results if "tier2_best_accuracy" in r]
        if t2_values:
            agg["tier2_best_accuracy_mean"] = safe_mean(t2_values)
            agg["tier2_best_accuracy_std"] = safe_std(t2_values)
            agg["tier2_best_accuracy_values"] = t2_values
        
        t2_layers = [r.get("tier2_best_layer", 0) for r in results if "tier2_best_layer" in r]
        if t2_layers:
            agg["tier2_best_layer_mode"] = int(max(set(t2_layers), key=t2_layers.count))
        
        # =====================================================================
        # IMPROVED TIER 3 AGGREGATION
        # =====================================================================
        
        # Basic Tier 3 metrics (mean recovery)
        t3_recovery = [r.get("tier3_mean_recovery") for r in results if r.get("tier3_mean_recovery") is not None]
        if t3_recovery:
            agg["tier3_mean_recovery_mean"] = safe_mean(t3_recovery)
            agg["tier3_mean_recovery_std"] = safe_std(t3_recovery)
        
        # Sparsity
        t3_sparsity = [r.get("tier3_sparsity") for r in results if r.get("tier3_sparsity") is not None]
        if t3_sparsity:
            agg["tier3_sparsity_mean"] = safe_mean(t3_sparsity)
            agg["tier3_sparsity_std"] = safe_std(t3_sparsity)
        
        # NEW: Robust recovery metrics (median-based)
        t3_recovery_median = [r.get("tier3_recovery_median") for r in results if r.get("tier3_recovery_median") is not None]
        if t3_recovery_median:
            agg["tier3_recovery_median_mean"] = safe_mean(t3_recovery_median)
            agg["tier3_recovery_median_std"] = safe_std(t3_recovery_median)
            agg["tier3_recovery_median_median"] = safe_median(t3_recovery_median)
            agg["tier3_recovery_median_values"] = t3_recovery_median
        
        # NEW: IQR
        t3_recovery_iqr = [r.get("tier3_recovery_iqr") for r in results if r.get("tier3_recovery_iqr") is not None]
        if t3_recovery_iqr:
            agg["tier3_recovery_iqr_mean"] = safe_mean(t3_recovery_iqr)
        
        # NEW: Valid pair rate
        t3_valid_rate = [r.get("tier3_valid_pair_rate") for r in results if r.get("tier3_valid_pair_rate") is not None]
        if t3_valid_rate:
            agg["tier3_valid_pair_rate_mean"] = safe_mean(t3_valid_rate)
            agg["tier3_valid_pair_rate_values"] = t3_valid_rate
        
        # NEW: Number of valid pairs
        t3_num_valid = [r.get("tier3_num_valid_pairs") for r in results if r.get("tier3_num_valid_pairs") is not None]
        if t3_num_valid:
            agg["tier3_num_valid_pairs_mean"] = safe_mean(t3_num_valid)
        
        # NEW: Mean gap
        t3_mean_gap = [r.get("tier3_mean_gap") for r in results if r.get("tier3_mean_gap") is not None]
        if t3_mean_gap:
            agg["tier3_mean_gap_mean"] = safe_mean(t3_mean_gap)
        
        # NEW: AURC
        t3_aurc = [r.get("tier3_aurc") for r in results if r.get("tier3_aurc") is not None]
        if t3_aurc:
            agg["tier3_aurc_mean"] = safe_mean(t3_aurc)
            agg["tier3_aurc_std"] = safe_std(t3_aurc)
        
        # NEW: Applicability tracking
        t3_applicable = [r.get("tier3_is_applicable", True) for r in results]
        agg["tier3_is_applicable"] = all(t3_applicable)  # True only if all seeds applicable
        agg["tier3_applicable_count"] = sum(1 for a in t3_applicable if a)
        
        # Get NA reason from first non-applicable result
        na_reasons = [r.get("tier3_na_reason") for r in results if r.get("tier3_na_reason")]
        agg["tier3_na_reason"] = na_reasons[0] if na_reasons else None
        
        # Aggregate layer recoveries
        layer_recoveries_all = [r.get("tier3_layer_recoveries", {}) for r in results]
        if layer_recoveries_all and layer_recoveries_all[0]:
            agg_layer_recoveries = {}
            for layer_key in layer_recoveries_all[0].keys():
                values = [lr.get(layer_key) for lr in layer_recoveries_all if lr.get(layer_key) is not None]
                if values:
                    agg_layer_recoveries[layer_key] = {
                        "mean": safe_mean(values),
                        "std": safe_std(values),
                        "median": safe_median(values),
                    }
            agg["tier3_layer_recoveries_aggregated"] = agg_layer_recoveries
        
        aggregated_results.append(agg)
    
    # Build final output
    aggregated = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_seeds": len(all_results),
            "seeds": seeds_used,
            "models": all_results[0]["metadata"]["models"],
            "tasks": all_results[0]["metadata"]["tasks"],
            "tiers": all_results[0]["metadata"]["tiers"],
            "num_positive": all_results[0]["metadata"].get("num_positive", 100),
            "num_negative": all_results[0]["metadata"].get("num_negative", 100),
            "num_pairs": all_results[0]["metadata"].get("num_pairs", 10),
        },
        "aggregated_results": aggregated_results,
        "raw_results": [r["results"] for r in all_results],
    }
    
    # Save aggregated results
    output_path = os.path.join(output_dir, "aggregated_results.json")
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nAggregated results saved to: {output_path}")
    
    # Also save a summary markdown
    save_summary_markdown(aggregated, output_dir)
    
    return aggregated


def save_summary_markdown(aggregated: Dict[str, Any], output_dir: str):
    """Save a human-readable summary with improved Tier-3 metrics."""
    
    md_path = os.path.join(output_dir, "summary.md")
    
    with open(md_path, "w") as f:
        f.write("# Multi-Seed Benchmark Results\n\n")
        f.write(f"**Generated**: {aggregated['metadata']['timestamp']}\n")
        f.write(f"**Seeds**: {aggregated['metadata']['seeds']}\n")
        f.write(f"**Models**: {aggregated['metadata']['models']}\n")
        f.write(f"**Tasks**: {aggregated['metadata']['tasks']}\n\n")
        
        # Main summary table
        f.write("## Summary Table\n\n")
        f.write("| Model | Task | Tier 1 | Tier 2 | T3 Median (IQR) | Valid% | AURC | Applicable |\n")
        f.write("|-------|------|--------|--------|-----------------|--------|------|------------|\n")
        
        for result in aggregated["aggregated_results"]:
            model = result["model"].split("/")[-1]  # Short name
            task = result["task"]
            
            t1 = f"{result.get('tier1_separation_mean', 0):.3f}±{result.get('tier1_separation_std', 0):.3f}"
            t2 = f"{result.get('tier2_best_accuracy_mean', 0):.3f}±{result.get('tier2_best_accuracy_std', 0):.3f}"
            
            # Tier 3 - use median if available, else mean
            if result.get('tier3_recovery_median_mean') is not None:
                t3_val = result['tier3_recovery_median_mean']
                t3_iqr = result.get('tier3_recovery_iqr_mean', 0)
                t3 = f"{t3_val:.3f} ({t3_iqr:.2f})"
            elif result.get('tier3_mean_recovery_mean') is not None:
                t3 = f"{result['tier3_mean_recovery_mean']:.3f}±{result.get('tier3_mean_recovery_std', 0):.3f}"
            else:
                t3 = "—"
            
            # Valid pair rate
            valid_rate = result.get('tier3_valid_pair_rate_mean')
            if valid_rate is not None:
                valid_str = f"{valid_rate:.0%}"
            else:
                valid_str = "—"
            
            # AURC
            aurc = result.get('tier3_aurc_mean')
            aurc_str = f"{aurc:.3f}" if aurc is not None else "—"
            
            # Applicability
            if result.get('tier3_is_applicable', True):
                applicable = "✓"
            else:
                na_reason = result.get('tier3_na_reason', 'NA')
                # Truncate reason
                applicable = f"NA"
            
            f.write(f"| {model} | {task} | {t1} | {t2} | {t3} | {valid_str} | {aurc_str} | {applicable} |\n")
        
        # Tier-3 details table for non-applicable cases
        na_cases = [r for r in aggregated["aggregated_results"] if not r.get('tier3_is_applicable', True)]
        if na_cases:
            f.write("\n## Tier-3 Non-Applicable Cases\n\n")
            f.write("| Model | Task | Reason | Valid Seeds |\n")
            f.write("|-------|------|--------|-------------|\n")
            for r in na_cases:
                model = r["model"].split("/")[-1]
                reason = r.get('tier3_na_reason', 'Unknown')[:50]
                valid_count = r.get('tier3_applicable_count', 0)
                f.write(f"| {model} | {r['task']} | {reason} | {valid_count}/{r['num_seeds']} |\n")
        
        # Detailed results
        f.write("\n## Detailed Results\n\n")
        
        for result in aggregated["aggregated_results"]:
            f.write(f"### {result['model']} x {result['task']}\n\n")
            f.write(f"- Architecture: {result['architecture']}\n")
            f.write(f"- Num layers: {result['num_layers']}\n")
            f.write(f"- Num seeds: {result['num_seeds']}\n")
            
            # Tier 3 details
            if result.get('tier3_valid_pair_rate_mean') is not None:
                f.write(f"- T3 Valid pair rate: {result['tier3_valid_pair_rate_mean']:.1%}\n")
            if result.get('tier3_mean_gap_mean') is not None:
                f.write(f"- T3 Mean gap: {result['tier3_mean_gap_mean']:.4f}\n")
            if not result.get('tier3_is_applicable', True):
                f.write(f"- **T3 Not Applicable**: {result.get('tier3_na_reason', 'Unknown')}\n")
            
            f.write("\n")
            
            if 'tier1_separation_values' in result:
                f.write(f"**Tier 1 values**: {result['tier1_separation_values']}\n\n")
            if 'tier2_best_accuracy_values' in result:
                f.write(f"**Tier 2 values**: {result['tier2_best_accuracy_values']}\n\n")
            if 'tier3_recovery_median_values' in result:
                f.write(f"**Tier 3 median values**: {result['tier3_recovery_median_values']}\n\n")
    
    print(f"Summary saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Run multi-seed benchmark')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
                        help='Seeds to use (default: 42 123 456 789 1000)')
    parser.add_argument('--output-dir', default='results/multiseed',
                        help='Output directory (default: results/multiseed)')
    parser.add_argument('--skip-run', action='store_true',
                        help='Skip running benchmarks, only aggregate existing results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks for each seed (unless skip-run)
    result_paths = []
    
    if not args.skip_run:
        for seed in args.seeds:
            try:
                result_path = run_benchmark_with_seed(
                    config_path=args.config,
                    seed=seed,
                    output_dir=args.output_dir
                )
                result_paths.append(result_path)
            except Exception as e:
                print(f"ERROR running seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Build expected paths
        for seed in args.seeds:
            result_path = os.path.join(args.output_dir, f"seed_{seed}", "benchmark_results.json")
            result_paths.append(result_path)
        print(f"Skipping benchmark runs, aggregating existing results...")
    
    # Aggregate results
    try:
        aggregated = aggregate_results(result_paths, args.output_dir)
        
        print(f"\n{'='*60}")
        print("MULTI-SEED BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"Seeds used: {aggregated['metadata']['seeds']}")
        print(f"Results: {args.output_dir}/aggregated_results.json")
        print(f"Summary: {args.output_dir}/summary.md")
        
    except Exception as e:
        print(f"\nERROR aggregating results: {e}")
        import traceback
        traceback.print_exc()
        
        # Debug: show what exists
        print(f"\nDebug: Contents of {args.output_dir}:")
        if os.path.exists(args.output_dir):
            for item in os.listdir(args.output_dir):
                item_path = os.path.join(args.output_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    for sub in os.listdir(item_path)[:3]:
                        print(f"      {sub}")
                else:
                    print(f"  📄 {item}")
        
        sys.exit(1)


if __name__ == '__main__':
    main()
