#!/usr/bin/env python3
"""
Gap Threshold Ablation Script

Demonstrates robustness of Tier-3 recovery to minimum gap threshold.
"""

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from mechinterp.core.registry import get_model, get_task, TIER_REGISTRY
import mechinterp.architectures
import mechinterp.tasks.sentiment
import mechinterp.tasks.nli


ENCODER_TASK_CHECKPOINTS = {
    ("bert-base-uncased", "sentiment"): "textattack/bert-base-uncased-SST-2",
    ("bert-base-uncased", "nli"):       "textattack/bert-base-uncased-MNLI",
    ("roberta-base", "sentiment"):      "textattack/roberta-base-SST-2",
    ("roberta-base", "nli"):            "textattack/roberta-base-MNLI",
}

def run_gap_ablation(
    model_name: str,
    task_name: str,
    thresholds: list = [0.0, 0.01, 0.05, 0.1],
    num_pairs: int = 20,
    seed: int = 42,
    device: str = "cpu"
):
    """Run Tier-3 with different gap thresholds."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load model
    task_type = 'classification' if task_name in ['sentiment', 'nli'] else 'features'
    
    if 'gpt' in model_name.lower():
        from mechinterp.architectures.decoder import create_decoder_wrapper
        model = create_decoder_wrapper(model_name, device=device, task='lm')
    else:
        from mechinterp.architectures.encoder import create_encoder_wrapper
        model = create_encoder_wrapper(ENCODER_TASK_CHECKPOINTS.get((model_name, task_name), model_name), device=device, task=task_type)
    
    _ = model.model
    
    # Load task
    task = get_task(task_name)
    if hasattr(task, 'load_data') and task._data is None:
        task.load_data()
    
    pos_examples = task.get_positive_examples(num_pairs * 2)
    neg_examples = task.get_negative_examples(num_pairs * 2, seed=seed)
    
    # Get Tier3 analyzer
    Tier3Analyzer = TIER_REGISTRY.get("tier3")
    
    results = {}
    for epsilon in thresholds:
        print(f"  Running with min_gap={epsilon}...")
        
        tier3 = Tier3Analyzer(model)
        tier3.min_gap = epsilon  # Set threshold
        
        try:
            result = tier3.analyze_with_multiple_examples(
                pos_examples[:num_pairs],
                neg_examples[:num_pairs],
                num_pairs=num_pairs
            )
            
            results[str(epsilon)] = {
                'recovery_median': getattr(result, 'recovery_median', None),
                'recovery_iqr': getattr(result, 'recovery_iqr', None),
                'valid_pair_rate': getattr(result, 'valid_pair_rate', None),
                'num_valid': getattr(result, 'num_valid_pairs', None),
                'mean_recovery': float(np.mean(list(result.component_effects.values()))),
            }
        except Exception as e:
            results[str(epsilon)] = {'error': str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['bert-base-uncased', 'gpt2'])
    parser.add_argument('--tasks', nargs='+', default=['sentiment', 'nli'])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.0, 0.01, 0.05, 0.1])
    parser.add_argument('--num-pairs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/gap_ablation.json')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    all_results = {}
    for model_name in args.models:
        for task_name in args.tasks:
            key = f"{model_name}_{task_name}"
            print(f"\nRunning: {key}")
            
            results = run_gap_ablation(
                model_name=model_name,
                task_name=task_name,
                thresholds=args.thresholds,
                num_pairs=args.num_pairs,
                seed=args.seed,
                device=device
            )
            all_results[key] = results
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("GAP ABLATION SUMMARY")
    print("="*60)
    
    for key, results in all_results.items():
        print(f"\n{key}:")
        print(f"  {'Threshold':<10} {'Median':<10} {'IQR':<10} {'Valid%':<10}")
        print(f"  {'-'*40}")
        for eps, data in results.items():
            if 'error' in data:
                print(f"  {eps:<10} ERROR: {data['error'][:30]}")
            else:
                median = data.get('recovery_median', 'N/A')
                iqr = data.get('recovery_iqr', 'N/A')
                valid = data.get('valid_pair_rate', 'N/A')
                
                median_str = f"{median:.3f}" if isinstance(median, float) else str(median)
                iqr_str = f"{iqr:.3f}" if isinstance(iqr, float) else str(iqr)
                valid_str = f"{valid:.0%}" if isinstance(valid, float) else str(valid)
                
                print(f"  {eps:<10} {median_str:<10} {iqr_str:<10} {valid_str:<10}")


if __name__ == '__main__':
    main()
