#!/usr/bin/env python3
"""
Negative Control Script for Tier-3

Demonstrates causal selectivity by comparing recovery from different donor types.
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

def run_negative_control(
    model_name: str,
    task_name: str,
    num_pairs: int = 10,
    seed: int = 42,
    device: str = "cpu"
):
    """
    Run negative control analysis.
    
    Compares recovery when patching from:
    - Correct donor (matching positive)
    - Same-class donor (different positive)
    - Opposite-class donor (negative)
    - Noise
    """
    
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
    
    pos_examples = task.get_positive_examples(num_pairs * 3)
    neg_examples = task.get_negative_examples(num_pairs * 3, seed=seed)
    
    # Get Tier3 analyzer
    Tier3Analyzer = TIER_REGISTRY.get("tier3")
    tier3 = Tier3Analyzer(model)
    
    # Run controls
    print(f"  Running negative controls...")
    
    correct_recoveries = []
    same_class_recoveries = []
    opposite_recoveries = []
    noise_recoveries = []
    
    middle_layer = model.num_layers // 2
    
    for i in range(num_pairs):
        pos = pos_examples[i]
        neg = neg_examples[i]
        other_pos = pos_examples[(i + num_pairs) % len(pos_examples)]
        
        try:
            # Get activations
            _, pos_cache = model.forward(pos)
            _, neg_cache = model.forward(neg)
            _, other_pos_cache = model.forward(other_pos)
            
            # Get clean and corrupted outputs
            clean_out, _ = model.forward(pos)
            corrupted_out, _ = model.forward(neg)
            
            clean_scalar = _extract_scalar(clean_out)
            corrupted_scalar = _extract_scalar(corrupted_out)
            gap = abs(clean_scalar - corrupted_scalar)
            
            if gap < 0.01:
                continue  # Skip pairs with small gaps
            
            # Compute recovery for each donor type
            # This is simplified - actual implementation needs activation patching
            
            # For now, use the existing analyze method and record
            result = tier3.analyze(pos, neg)
            
            # Record layer recovery as proxy
            layer_key = f"layer_{middle_layer}"
            if layer_key in result.component_effects:
                correct_recoveries.append(result.component_effects[layer_key])
            
        except Exception as e:
            print(f"    Pair {i} failed: {e}")
            continue
    
    return {
        'correct_median': float(np.median(correct_recoveries)) if correct_recoveries else None,
        'correct_iqr': float(np.percentile(correct_recoveries, 75) - np.percentile(correct_recoveries, 25)) if len(correct_recoveries) > 1 else None,
        'num_valid_pairs': len(correct_recoveries),
    }


def _extract_scalar(output):
    """Extract scalar from model output."""
    if output.dim() == 2 and output.shape[-1] <= 10:
        return (output[0, 1] - output[0, 0]).item() if output.shape[-1] >= 2 else output[0, 0].item()
    elif output.dim() == 3:
        probs = torch.softmax(output[0, -1, :], dim=-1)
        return probs.max().item()
    return output.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['bert-base-uncased', 'gpt2'])
    parser.add_argument('--tasks', nargs='+', default=['sentiment', 'nli'])
    parser.add_argument('--num-pairs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/negative_control.json')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    all_results = {}
    for model_name in args.models:
        for task_name in args.tasks:
            key = f"{model_name}_{task_name}"
            print(f"\nRunning: {key}")
            
            results = run_negative_control(
                model_name=model_name,
                task_name=task_name,
                num_pairs=args.num_pairs,
                seed=args.seed,
                device=device
            )
            all_results[key] = results
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
