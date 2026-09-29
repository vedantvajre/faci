#!/usr/bin/env python3
"""
Benchmark Runner - Compatible with mechinterp Registry

Uses mechinterp's registry system for models, tasks, and tiers.
"""

import argparse
import json
import os
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import mechinterp components
from mechinterp.core.registry import (
    get_model, get_task, 
    MODEL_REGISTRY, TASK_REGISTRY, TIER_REGISTRY,
    list_models, list_tasks
)
from mechinterp.core.base import Architecture

# Import architectures to trigger registration
import mechinterp.architectures
# Import tasks to trigger registration  
import mechinterp.tasks.sentiment
import mechinterp.tasks.nli
import mechinterp.tasks.factual_recall
import mechinterp.tasks.qa
# Import tiers to trigger registration
import mechinterp.tiers.tier1
import mechinterp.tiers.tier2
import mechinterp.tiers.tier3

# Try to import output functions
try:
    from mechinterp.output_functions import get_output_fn, OUTPUT_FN_REGISTRY
    HAS_OUTPUT_FN = True
except ImportError:
    HAS_OUTPUT_FN = False
    print("Note: output_functions.py not found, using default Tier 3 behavior")


ENCODER_TASK_CHECKPOINTS = {
    ("bert-base-uncased", "sentiment"): "textattack/bert-base-uncased-SST-2",
    ("bert-base-uncased", "nli"):       "textattack/bert-base-uncased-MNLI",
    ("roberta-base", "sentiment"):      "textattack/roberta-base-SST-2",
    ("roberta-base", "nli"):            "textattack/roberta-base-MNLI",
}

def get_task_type(task_name: str) -> str:
    """Map task name to task type for model loading."""
    if task_name in ['sentiment', 'nli']:
        return 'classification'
    elif task_name == 'factual_recall':
        return 'mlm'  # For encoders, 'lm' for decoders
    elif task_name == 'qa':
        return 'features'
    return 'classification'


def run_single_benchmark(
    model_name: str,
    task_name: str,
    tiers: List[int],
    config: Dict[str, Any],
    device: str
) -> Dict[str, Any]:
    """Run benchmark for a single model-task combination."""
    
    print(f"\n{'='*60}")
    print(f"Running: {model_name} x {task_name}")
    print(f"Tiers: {tiers}")
    print(f"{'='*60}")
    
    # Get seed from config
    seed = config.get('seed', 42)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Determine task type for model loading
    task_type = get_task_type(task_name)
    model_load_name = ENCODER_TASK_CHECKPOINTS.get((model_name, task_name), model_name)
    
    # Load model using registry
    print(f"  Loading model: {model_load_name} (task={task_type})")
    try:
        # Check if model is in registry
        if model_load_name in MODEL_REGISTRY:
            model = get_model(model_load_name, device=device, task=task_type)
        else:
            # Try to create wrapper based on model name
            if any(k in model_load_name.lower() for k in ['gpt','pythia','qwen','llama','opt','mistral']):
                from mechinterp.architectures.decoder import create_decoder_wrapper
                model = create_decoder_wrapper(model_load_name, device=device, task='lm')
            elif 't5' in model_load_name.lower() or 'bart' in model_load_name.lower():
                from mechinterp.architectures.enc_dec import create_enc_dec_wrapper
                model = create_enc_dec_wrapper(model_load_name, device=device, task='seq2seq')
            else:
                from mechinterp.architectures.encoder import create_encoder_wrapper
                model = create_encoder_wrapper(model_load_name, device=device, task=task_type)
        
        # Ensure model is loaded
        _ = model.model  # This triggers load() if not already loaded
        
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return {'model': model_name, 'task': task_name, 'error': str(e)}
    
    # Get architecture
    architecture = model.architecture.value
    print(f"  Architecture: {architecture}, Layers: {model.num_layers}")
    
    # Load task
    print(f"  Loading task: {task_name}")
    try:
        task = get_task(task_name)
        # Ensure task data is loaded
        if hasattr(task, 'load_data') and task._data is None:
            task.load_data()
    except Exception as e:
        print(f"  ERROR loading task: {e}")
        import traceback
        traceback.print_exc()
        return {'model': model_name, 'task': task_name, 'error': str(e)}
    
    results = {
        'model': model_name,
        'task': task_name,
        'architecture': architecture,
        'num_layers': model.num_layers,
    }
    results['tier3_intervention'] = 'readout-pos-v3-taskfix'
    results['checkpoint'] = model_load_name
    
    # Get examples
    num_positive = config.get('num_positive', 100)
    num_negative = config.get('num_negative', 100)
    num_pairs = config.get('num_pairs', 10)
    
    positive_examples = task.get_positive_examples(num_positive, seed=seed)
    negative_examples = task.get_negative_examples(num_negative, seed=seed)
    
    print(f"  Got {len(positive_examples)} positive, {len(negative_examples)} negative examples")
    
    # Run Tier 1
    if 1 in tiers:
        print(f"  Running Tier 1...")
        try:
            Tier1Analyzer = TIER_REGISTRY.get("tier1")
            tier1 = Tier1Analyzer(model)  # Pass model wrapper only
            from tier1_heldout_analyze import heldout_tier1
            t1h = heldout_tier1(tier1, positive_examples, negative_examples, seed=seed, n_perm=200)
            results['tier1_separation'] = t1h['tier1_separation']
            results['tier1_perm_null_mean'] = t1h['tier1_perm_null_mean']
            results['tier1_perm_null_p95'] = t1h['tier1_perm_null_p95']
            print(f"    Tier 1 held-out: {t1h['tier1_separation']:.4f} (null {t1h['tier1_perm_null_mean']:.3f})")
        except Exception as e:
            print(f"  Tier 1 failed: {e}")
            import traceback
            traceback.print_exc()
            results['tier1_separation'] = 0
    
    # Run Tier 2
    if 2 in tiers:
        print(f"  Running Tier 2...")
        try:
            Tier2Analyzer = TIER_REGISTRY.get("tier2")
            tier2 = Tier2Analyzer(model)  # Pass model wrapper only
            tier2_result = tier2.analyze(positive_examples, negative_examples)
            results['tier2_best_layer'] = tier2_result.best_layer
            results['tier2_best_accuracy'] = tier2_result.best_accuracy
            from tier2_heldout_analyze import heldout_tier2
            _t2h = heldout_tier2(tier2, positive_examples, negative_examples, seed=seed, paired=(task_name in ('qa','factual_recall')))
            results['tier2_best_accuracy'] = _t2h['tier2_best_accuracy']
            results['tier2_best_layer'] = _t2h['tier2_best_layer']
            results['tier2_protocol'] = _t2h['protocol']

            results['tier2_emergence_layer'] = tier2_result.emergence_layer
            results['tier2_layer_accuracies'] = tier2_result.layer_accuracies
            print(f"    Tier 2 best: {tier2_result.best_accuracy:.4f} at layer {tier2_result.best_layer}")
        except Exception as e:
            print(f"  Tier 2 failed: {e}")
            import traceback
            traceback.print_exc()
            results['tier2_best_layer'] = 0
            results['tier2_best_accuracy'] = 0
    
    # Run Tier 3
    if 3 in tiers:
        print(f"  Running Tier 3...")
        try:
            Tier3Analyzer = TIER_REGISTRY.get("tier3")
            tier3 = Tier3Analyzer(model)  # Pass model wrapper only
            
            # Check if config wants baselines
            run_baselines = config.get('run_baselines', False)
            
            # Use multiple examples for better results
            tier3_result = tier3.analyze_with_multiple_examples(
                positive_examples[:num_pairs],
                negative_examples[:num_pairs],
                num_pairs=num_pairs,
            )
            
            # Standard fields
            results['tier3_layer_recoveries'] = tier3_result.component_effects
            results['tier3_circuit_size'] = tier3_result.circuit_size
            results['tier3_sparsity'] = tier3_result.sparsity
            
            # Calculate summary metrics (filter NaN)
            recoveries = [v for v in tier3_result.component_effects.values() 
                          if v is not None and not np.isnan(v)]
            
            if recoveries:
                results['tier3_mean_recovery'] = float(np.mean(recoveries))
                results['tier3_std_recovery'] = float(np.std(recoveries))
                results['tier3_median_recovery'] = float(np.median(recoveries))
                results['tier3_iqr_recovery'] = float(
                    np.percentile(recoveries, 75) - np.percentile(recoveries, 25)
                )
                results['tier3_max_recovery'] = float(np.max(recoveries))
                results['tier3_min_recovery'] = float(np.min(recoveries))
            
            # New robust stats (if available from Tier3ResultExtended)
            if hasattr(tier3_result, 'recovery_median'):
                results['tier3_recovery_median'] = tier3_result.recovery_median
                results['tier3_recovery_iqr'] = tier3_result.recovery_iqr
                results['tier3_valid_pair_rate'] = tier3_result.valid_pair_rate
                results['tier3_num_valid_pairs'] = tier3_result.num_valid_pairs
                results['tier3_mean_gap'] = tier3_result.mean_gap
                results['tier3_aurc'] = tier3_result.aurc
                results['tier3_is_applicable'] = tier3_result.is_applicable
                results['tier3_na_reason'] = tier3_result.na_reason
            
            # Print summary
            if recoveries:
                median_str = f", median: {results.get('tier3_median_recovery', 0):.4f}" if 'tier3_median_recovery' in results else ""
                valid_str = f", valid: {results.get('tier3_valid_pair_rate', 1.0):.0%}" if 'tier3_valid_pair_rate' in results else ""
                print(f"    Tier 3 mean: {results['tier3_mean_recovery']:.4f}{median_str}{valid_str}")
            
            # Run baselines if requested
            if run_baselines and hasattr(tier3, 'run_randomization_baselines'):
                print(f"    Running randomization baselines...")
                baselines = tier3.run_randomization_baselines(
                    positive_examples[:num_pairs],
                    negative_examples[:num_pairs],
                    num_pairs=min(5, num_pairs),
                )
                results['tier3_baseline_same_class'] = baselines.get('same_class_recovery')
                results['tier3_baseline_opposite_class'] = baselines.get('opposite_class_recovery')
                results['tier3_baseline_noise'] = baselines.get('noise_recovery')
                print(f"    Baselines - same: {baselines.get('same_class_recovery', 0):.3f}, "
                      f"opposite: {baselines.get('opposite_class_recovery', 0):.3f}, "
                      f"noise: {baselines.get('noise_recovery', 0):.3f}")
            
        except Exception as e:
            print(f"  Tier 3 failed: {e}")
            import traceback
            traceback.print_exc()
            results['tier3_layer_recoveries'] = {}
            results['tier3_sparsity'] = 0
    return results


def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run full benchmark from config."""
    
    models = config.get('models', ['bert-base-uncased'])
    tasks = config.get('tasks', ['sentiment'])
    tiers = config.get('tiers', [1, 2, 3])
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle MPS for Apple Silicon
    if device == 'cuda' and not torch.cuda.is_available():
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    print(f"Available models in registry: {list_models()}")
    print(f"Available tasks in registry: {list_tasks()}")
    
    all_results = []
    
    for model_name in models:
        for task_name in tasks:
            try:
                result = run_single_benchmark(
                    model_name=model_name,
                    task_name=task_name,
                    tiers=tiers,
                    config=config,
                    device=device
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: {model_name} x {task_name} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'model': model_name,
                    'task': task_name,
                    'error': str(e)
                })
    
    # Build output
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'models': models,
            'tasks': tasks,
            'tiers': tiers,
            'num_positive': config.get('num_positive', 100),
            'num_negative': config.get('num_negative', 100),
            'num_pairs': config.get('num_pairs', 10),
            'seed': config.get('seed', 42),
        },
        'results': all_results
    }
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Run mechanistic interpretability benchmark')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Run benchmark
    results = run_benchmark(config)
    
    # CRITICAL: Use output_dir from config
    output_dir = config.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'benchmark_results.json')
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    for result in results['results']:
        if 'error' in result:
            print(f"  {result['model']} x {result['task']}: ERROR - {result['error']}")
            continue
        print(f"  {result['model']} x {result['task']}:")
        if 'tier1_separation' in result:
            print(f"    Tier 1: {result['tier1_separation']:.4f}")
        if 'tier2_best_accuracy' in result:
            print(f"    Tier 2: {result['tier2_best_accuracy']:.4f} (L{result['tier2_best_layer']})")
        if 'tier3_mean_recovery' in result:
            print(f"    Tier 3: {result['tier3_mean_recovery']:.4f} (sparsity: {result.get('tier3_sparsity', 0):.2f})")


if __name__ == '__main__':
    main()
