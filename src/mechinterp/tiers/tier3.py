"""
Tier 3: Activation Patching / Circuit Analysis (Improved)

Key improvements:
1. Gap-based filtering with configurable threshold
2. Valid-pair rate tracking
3. Robust statistics (median, IQR, trimmed mean)
4. Randomization baselines for causal specificity
5. Area Under Recovery Curve (AURC) metric
6. Explicit NA handling (not NaN)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from mechinterp.core.base import (
    BaseModelWrapper,
    BaseTierAnalyzer,
    ExtractionPosition,
    PatchTarget,
    Tier3Result,
    ActivationCache,
)
from mechinterp.core.registry import register_tier

logger = logging.getLogger(__name__)


@dataclass
class Tier3ResultExtended(Tier3Result):
    """Extended Tier 3 result with robust statistics."""
    
    # Robust statistics
    recovery_median: float = 0.0
    recovery_iqr: float = 0.0  # Interquartile range
    recovery_trimmed_mean: float = 0.0  # 10% trimmed
    
    # Validity metrics
    valid_pair_rate: float = 1.0
    num_valid_pairs: int = 0
    num_total_pairs: int = 0
    mean_gap: float = 0.0
    
    # Per-layer robust stats
    layer_recovery_medians: Dict[str, float] = field(default_factory=dict)
    layer_recovery_iqrs: Dict[str, float] = field(default_factory=dict)
    
    # AURC metrics
    aurc: float = 0.0  # Area under recovery curve
    top_k_recovery: float = 0.0  # Sum of top-k layers
    
    # Baseline comparisons (for causal specificity)
    random_same_class_recovery: Optional[float] = None
    random_opposite_class_recovery: Optional[float] = None
    noise_baseline_recovery: Optional[float] = None
    
    # Applicability
    is_applicable: bool = True
    na_reason: Optional[str] = None


def compute_recovery(
    patched: float,
    corrupted: float, 
    clean: float,
    min_gap: float = 0.01,
) -> Tuple[float, bool]:
    """
    Compute recovery with validity flag.
    
    Returns:
        (recovery_value, is_valid)
    """
    gap = clean - corrupted
    
    if abs(gap) < min_gap:
        return float('nan'), False
    
    recovery = (patched - corrupted) / gap
    return recovery, True


def compute_robust_stats(values: List[float]) -> Dict[str, float]:
    """Compute robust statistics from a list of values."""
    # Filter NaN
    valid = [v for v in values if not np.isnan(v)]
    
    if not valid:
        return {
            "mean": float('nan'),
            "std": float('nan'),
            "median": float('nan'),
            "iqr": float('nan'),
            "trimmed_mean": float('nan'),
            "min": float('nan'),
            "max": float('nan'),
            "n_valid": 0,
            "n_total": len(values),
        }
    
    arr = np.array(valid)
    
    # Trimmed mean (10% from each end)
    trim_n = max(1, len(arr) // 10)
    if len(arr) > 2 * trim_n:
        sorted_arr = np.sort(arr)
        trimmed = sorted_arr[trim_n:-trim_n]
        trimmed_mean = np.mean(trimmed)
    else:
        trimmed_mean = np.mean(arr)
    
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "median": np.median(arr),
        "iqr": np.percentile(arr, 75) - np.percentile(arr, 25),
        "trimmed_mean": trimmed_mean,
        "min": np.min(arr),
        "max": np.max(arr),
        "n_valid": len(valid),
        "n_total": len(values),
    }


def compute_aurc(layer_recoveries: Dict[int, float]) -> float:
    """
    Compute Area Under Recovery Curve.
    
    Integrates recovery over layers (normalized by num layers).
    """
    if not layer_recoveries:
        return float('nan')
    
    valid_recoveries = [r for r in layer_recoveries.values() if not np.isnan(r)]
    if not valid_recoveries:
        return float('nan')
    
    # Simple trapezoidal integration normalized by number of layers
    sorted_layers = sorted(layer_recoveries.keys())
    total = 0.0
    valid_count = 0
    
    for layer in sorted_layers:
        r = layer_recoveries[layer]
        if not np.isnan(r):
            total += r
            valid_count += 1
    
    return total / valid_count if valid_count > 0 else float('nan')


def compute_top_k_recovery(layer_recoveries: Dict[int, float], k: int = 3) -> float:
    """Sum of top-k layer recoveries."""
    valid = [(l, r) for l, r in layer_recoveries.items() if not np.isnan(r)]
    if not valid:
        return float('nan')
    
    sorted_by_recovery = sorted(valid, key=lambda x: x[1], reverse=True)
    top_k = sorted_by_recovery[:k]
    return sum(r for _, r in top_k)


@register_tier("tier3")
class Tier3Analyzer(BaseTierAnalyzer):
    """
    Improved Tier 3 Analyzer with:
    - Gap-based filtering
    - Robust statistics
    - Randomization baselines
    - AURC metrics
    """

    def __init__(
        self,
        model: BaseModelWrapper,
        patch_target: PatchTarget = PatchTarget.LAYER,
        min_gap: float = 0.01,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.patch_target = patch_target
        self.min_gap = min_gap
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._feature_direction: Optional[Tensor] = None

    @property
    def tier(self) -> int:
        return 3

    def _clear_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def set_feature_direction(self, direction: Tensor) -> None:
        """Set feature direction for output scalar computation."""
        self._feature_direction = direction.detach().cpu()

    def _compute_output_scalar(
        self,
        output: Tensor,
        output_fn: Optional[Callable] = None,
    ) -> float:
        """Extract scalar from model output."""
        if output_fn:
            return output_fn(output)
        
        arch = self.model.architecture.value
        is_autoregressive = arch in ("decoder", "encoder_decoder")
        token_idx = -1 if is_autoregressive else 0
        
        if output.dim() == 2 and output.shape[-1] <= 10:
            # Classification logits
            if output.shape[-1] >= 2:
                return (output[0, 1] - output[0, 0]).item()
            return output[0, 0].item()
        
        elif output.dim() == 3:
            if output.shape[-1] > 1000:
                # LM logits - use top probability
                token_logits = output[0, token_idx, :]
                probs = torch.softmax(token_logits, dim=-1)
                return probs.max().item()
            else:
                # Hidden states
                hidden = output[0, token_idx, :]
                if self._feature_direction is not None:
                    direction = self._feature_direction.to(hidden.device).float()
                    direction = direction / (torch.norm(direction) + 1e-8)
                    return torch.dot(hidden.float(), direction).item()
                return hidden.float().mean().item()
        
        elif output.dim() == 2:
            hidden = output[0, :]
            if self._feature_direction is not None:
                direction = self._feature_direction.to(hidden.device).float()
                direction = direction / (torch.norm(direction) + 1e-8)
                return torch.dot(hidden.float(), direction).item()
            return hidden.float().mean().item()
        
        return output.float().mean().item()

    def _get_baseline_output(
        self,
        text: str,
        output_fn: Optional[Callable] = None,
    ) -> float:
        """Get baseline model output for a text."""
        with torch.no_grad():
            output, _ = self.model.forward(text)
        return self._compute_output_scalar(output, output_fn)

    def _run_with_patch(
        self,
        text: str,
        patch_layer: int,
        patch_activation: Tensor,
        component: str = "layer",
        output_fn: Optional[Callable] = None,
    ) -> float:
        """Run model with patched activation."""
        self._clear_hooks()
        
        if component == "layer":
            modules = self.model._get_layer_modules()
        elif component == "attn":
            modules = self.model._get_attention_modules()
        elif component == "mlp":
            modules = self.model._get_mlp_modules()
        else:
            raise ValueError(f"Unknown component: {component}")
        
        if patch_layer >= len(modules):
            raise ValueError(f"Layer {patch_layer} out of range")
        
        module = modules[patch_layer]
        
        def patch_hook(mod, input, output):
            if isinstance(output, tuple):
                orig = output[0]
            else:
                orig = output
            
            # read-out-position patch; branch by class hierarchy (isinstance), not name
            from mechinterp.architectures.decoder import DecoderWrapper as _DW
            from mechinterp.architectures.enc_dec import EncoderDecoderWrapper as _EDW
            patch_src = patch_activation.to(orig.device)
            if isinstance(self.model, (_DW, _EDW)):
                _dp = orig.shape[1] - 1; _sp = patch_src.shape[1] - 1
            else:
                _dp = 0; _sp = 0
            patch = orig.clone()
            patch[:, _dp, :] = patch_src[:, _sp, :]
            
            if isinstance(output, tuple):
                return (patch,) + output[1:]
            return patch
        
        hook = module.register_forward_hook(patch_hook)
        self._hooks.append(hook)
        
        inputs = self.model._tokenize(text)
        
        if self.model.architecture.value == "encoder_decoder":
            batch_size = inputs["input_ids"].shape[0]
            config = self.model.model.config
            start_token_id = getattr(config, "decoder_start_token_id", 0)
            if start_token_id is None:
                start_token_id = getattr(self.model.tokenizer, "pad_token_id", 0) or 0
            inputs["decoder_input_ids"] = torch.full(
                (batch_size, 1), start_token_id,
                dtype=torch.long, device=inputs["input_ids"].device
            )
        
        with torch.no_grad():
            outputs = self.model.model(**inputs, output_hidden_states=True)
        
        self._clear_hooks()
        
        if hasattr(outputs, "logits"):
            output = outputs.logits
        else:
            output = outputs.last_hidden_state
        
        return self._compute_output_scalar(output, output_fn)

    def patch_layer(
        self,
        clean_text: str,
        corrupted_text: str,
        layer: int,
        component: str = "layer",
        output_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Patch activations at a specific layer."""
        clean_output = self._get_baseline_output(clean_text, output_fn)
        
        with torch.no_grad():
            _, clean_cache = self.model.forward(clean_text)
        clean_activation = clean_cache.hidden_states.get(f"layer_{layer}")
        
        corrupted_output = self._get_baseline_output(corrupted_text, output_fn)
        
        if clean_activation is not None:
            patched_output = self._run_with_patch(
                corrupted_text, layer, clean_activation, component, output_fn
            )
        else:
            patched_output = corrupted_output
        
        gap = clean_output - corrupted_output
        recovery, is_valid = compute_recovery(
            patched_output, corrupted_output, clean_output, self.min_gap
        )
        
        return {
            "clean_output": clean_output,
            "corrupted_output": corrupted_output,
            "patched_output": patched_output,
            "recovery": recovery,
            "gap": gap,
            "is_valid": is_valid,
        }

    def patch_all_layers(
        self,
        clean_text: str,
        corrupted_text: str,
        component: str = "layer",
        output_fn: Optional[Callable] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Patch each layer and measure effect."""
        results = {}
        num_layers = self.model.num_layers
        
        for layer in range(num_layers):
            results[layer] = self.patch_layer(
                clean_text, corrupted_text, layer, component, output_fn
            )
        
        return results

    def analyze_with_multiple_examples(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        num_pairs: int = 10,
        **kwargs
    ) -> Tier3ResultExtended:
        """
        Run Tier 3 with multiple example pairs and robust aggregation.
        """
        component = kwargs.get("component", "layer")
        threshold = kwargs.get("threshold", 0.5)
        output_fn = kwargs.get("output_fn", None)
        
        all_recoveries: Dict[int, List[float]] = {}
        all_gaps: List[float] = []
        valid_pair_count = 0
        
        pairs = min(num_pairs, len(positive_examples), len(negative_examples))
        
        for i in tqdm(range(pairs), desc="Patching pairs"):
            clean = positive_examples[i]
            corrupted = negative_examples[i]
            
            # Check gap first
            clean_output = self._get_baseline_output(clean, output_fn)
            corrupted_output = self._get_baseline_output(corrupted, output_fn)
            gap = abs(clean_output - corrupted_output)
            all_gaps.append(gap)
            
            if gap < self.min_gap:
                # Skip pair but track it
                continue
            
            valid_pair_count += 1
            
            results = self.patch_all_layers(clean, corrupted, component, output_fn)
            
            for layer, r in results.items():
                if layer not in all_recoveries:
                    all_recoveries[layer] = []
                all_recoveries[layer].append(r["recovery"])
        
        # Compute per-layer robust stats
        layer_stats = {}
        layer_medians = {}
        layer_iqrs = {}
        layer_means = {}
        
        for layer in sorted(all_recoveries.keys()):
            stats = compute_robust_stats(all_recoveries[layer])
            layer_stats[f"layer_{layer}"] = stats
            layer_medians[f"layer_{layer}"] = stats["median"]
            layer_iqrs[f"layer_{layer}"] = stats["iqr"]
            layer_means[f"layer_{layer}"] = stats["mean"]
        
        # Compute overall robust stats
        all_layer_medians = [m for m in layer_medians.values() if not np.isnan(m)]
        overall_stats = compute_robust_stats(all_layer_medians)
        
        # Compute AURC and top-k
        layer_recovery_dict = {int(k.split("_")[1]): v for k, v in layer_medians.items()}
        aurc = compute_aurc(layer_recovery_dict)
        top_k = compute_top_k_recovery(layer_recovery_dict, k=3)
        
        # Find important layers (using median)
        important = [
            k for k, m in layer_medians.items()
            if not np.isnan(m) and m >= threshold
        ]
        
        # Validity metrics
        valid_pair_rate = valid_pair_count / pairs if pairs > 0 else 0.0
        mean_gap = np.mean(all_gaps) if all_gaps else 0.0
        
        # Determine if results are applicable
        is_applicable = valid_pair_count >= 3  # Need at least 3 valid pairs
        na_reason = None
        if not is_applicable:
            na_reason = f"Insufficient valid pairs ({valid_pair_count}/{pairs})"
        
        return Tier3ResultExtended(
            # Standard fields
            component_effects=layer_means,
            important_components=important,
            circuit_size=len(important),
            total_components=self.model.num_layers,
            sparsity=1.0 - (len(important) / self.model.num_layers) if self.model.num_layers > 0 else 0.0,
            
            # Robust stats
            recovery_median=overall_stats["median"],
            recovery_iqr=overall_stats["iqr"],
            recovery_trimmed_mean=overall_stats["trimmed_mean"],
            
            # Validity
            valid_pair_rate=valid_pair_rate,
            num_valid_pairs=valid_pair_count,
            num_total_pairs=pairs,
            mean_gap=mean_gap,
            
            # Per-layer robust
            layer_recovery_medians=layer_medians,
            layer_recovery_iqrs=layer_iqrs,
            
            # AURC
            aurc=aurc,
            top_k_recovery=top_k,
            
            # Applicability
            is_applicable=is_applicable,
            na_reason=na_reason,
        )

    def run_randomization_baselines(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        num_pairs: int = 5,
        **kwargs
    ) -> Dict[str, float]:
        """
        Run randomization baselines for causal specificity.
        
        Tests:
        1. Patch from random same-class example
        2. Patch from opposite-class example  
        3. Patch with noise/mean activation
        """
        import random
        
        component = kwargs.get("component", "layer")
        output_fn = kwargs.get("output_fn", None)
        
        same_class_recoveries = []
        opposite_class_recoveries = []
        noise_recoveries = []
        
        pairs = min(num_pairs, len(positive_examples) - 1, len(negative_examples))
        
        for i in range(pairs):
            clean = positive_examples[i]
            corrupted = negative_examples[i]
            
            # Check gap
            clean_output = self._get_baseline_output(clean, output_fn)
            corrupted_output = self._get_baseline_output(corrupted, output_fn)
            gap = abs(clean_output - corrupted_output)
            
            if gap < self.min_gap:
                continue
            
            # Get clean activation from a DIFFERENT positive example
            other_positives = [p for j, p in enumerate(positive_examples) if j != i]
            if other_positives:
                random_same = random.choice(other_positives)
                with torch.no_grad():
                    _, same_cache = self.model.forward(random_same)
                
                # Test on middle layer
                mid_layer = self.model.num_layers // 2
                same_activation = same_cache.hidden_states.get(f"layer_{mid_layer}")
                
                if same_activation is not None:
                    patched = self._run_with_patch(
                        corrupted, mid_layer, same_activation, component, output_fn
                    )
                    rec, valid = compute_recovery(patched, corrupted_output, clean_output, self.min_gap)
                    if valid:
                        same_class_recoveries.append(rec)
            
            # Patch from opposite class
            random_opposite = random.choice(negative_examples[:i] + negative_examples[i+1:]) if len(negative_examples) > 1 else negative_examples[0]
            with torch.no_grad():
                _, opp_cache = self.model.forward(random_opposite)
            
            opp_activation = opp_cache.hidden_states.get(f"layer_{mid_layer}")
            if opp_activation is not None:
                patched = self._run_with_patch(
                    corrupted, mid_layer, opp_activation, component, output_fn
                )
                rec, valid = compute_recovery(patched, corrupted_output, clean_output, self.min_gap)
                if valid:
                    opposite_class_recoveries.append(rec)
            
            # Patch with noise (Gaussian with same mean/std as clean activation)
            with torch.no_grad():
                _, clean_cache = self.model.forward(clean)
            clean_act = clean_cache.hidden_states.get(f"layer_{mid_layer}")
            
            if clean_act is not None:
                noise = torch.randn_like(clean_act) * clean_act.std() + clean_act.mean()
                patched = self._run_with_patch(
                    corrupted, mid_layer, noise, component, output_fn
                )
                rec, valid = compute_recovery(patched, corrupted_output, clean_output, self.min_gap)
                if valid:
                    noise_recoveries.append(rec)
        
        return {
            "same_class_recovery": np.median(same_class_recoveries) if same_class_recoveries else float('nan'),
            "opposite_class_recovery": np.median(opposite_class_recoveries) if opposite_class_recoveries else float('nan'),
            "noise_recovery": np.median(noise_recoveries) if noise_recoveries else float('nan'),
            "n_same_class": len(same_class_recoveries),
            "n_opposite_class": len(opposite_class_recoveries),
            "n_noise": len(noise_recoveries),
        }

    def analyze(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        **kwargs
    ) -> Tier3ResultExtended:
        """
        Full Tier 3 analysis with robust statistics and baselines.
        """
        num_pairs = kwargs.get("num_pairs", 10)
        run_baselines = kwargs.get("run_baselines", False)
        
        # Main analysis
        result = self.analyze_with_multiple_examples(
            positive_examples, negative_examples, num_pairs, **kwargs
        )
        
        # Optional: Run baselines
        if run_baselines and result.is_applicable:
            baselines = self.run_randomization_baselines(
                positive_examples, negative_examples, 
                num_pairs=min(5, num_pairs), **kwargs
            )
            result.random_same_class_recovery = baselines["same_class_recovery"]
            result.random_opposite_class_recovery = baselines["opposite_class_recovery"]
            result.noise_baseline_recovery = baselines["noise_recovery"]
        
        return result
