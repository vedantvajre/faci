"""
Utility functions for mechinterp.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("mechinterp")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.flatten().float()
    b = b.flatten().float()
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()


def compute_direction(
    positive_activations: Tensor,
    negative_activations: Tensor,
    normalize: bool = True,
) -> Tensor:
    """
    Compute feature direction from positive and negative examples.
    
    Args:
        positive_activations: (N, hidden_size) tensor
        negative_activations: (M, hidden_size) tensor
        normalize: Whether to normalize the direction
        
    Returns:
        Direction vector of shape (hidden_size,)
    """
    pos_mean = positive_activations.mean(dim=0)
    neg_mean = negative_activations.mean(dim=0)
    direction = pos_mean - neg_mean
    
    if normalize:
        direction = direction / torch.norm(direction)
    
    return direction


def project_onto_direction(
    activations: Tensor,
    direction: Tensor,
) -> Tensor:
    """
    Project activations onto a direction.
    
    Args:
        activations: (batch, hidden_size) tensor
        direction: (hidden_size,) tensor (should be normalized)
        
    Returns:
        Projection scores of shape (batch,)
    """
    if activations.dim() == 1:
        activations = activations.unsqueeze(0)
    
    direction = direction / torch.norm(direction)
    return torch.matmul(activations, direction)


def compute_logit_diff(
    logits: Tensor,
    positive_idx: int = 1,
    negative_idx: int = 0,
) -> Tensor:
    """
    Compute logit difference (positive - negative).
    
    Args:
        logits: (batch, num_classes) tensor
        positive_idx: Index of positive class
        negative_idx: Index of negative class
        
    Returns:
        Logit differences of shape (batch,)
    """
    return logits[:, positive_idx] - logits[:, negative_idx]


def compute_recovery(
    patched: float,
    corrupted: float,
    clean: float,
    min_gap: float = 0.01,
) -> float:
    """
    Compute recovery metric for activation patching.
    
    recovery = (patched - corrupted) / (clean - corrupted)
    
    Recovery measures how much of the clean-corrupted gap is recovered
    by patching:
    - 0.0 = no effect (patched == corrupted)
    - 1.0 = full recovery (patched == clean)
    - > 1.0 = overcorrection (patched beyond clean)
    - < 0.0 = inverse effect (patched further from clean)
    
    Args:
        patched: Output after patching
        corrupted: Corrupted/baseline output (before patching)
        clean: Clean/target output (from source)
        min_gap: Minimum gap for reliable recovery computation
        
    Returns:
        Recovery score, or NaN if gap is too small for reliable measurement
    """
    gap = clean - corrupted
    
    # If gap is too small, recovery metric is unreliable
    # Return NaN instead of an arbitrary value
    if abs(gap) < min_gap:
        return float('nan')
    
    recovery = (patched - corrupted) / gap
    
    # No artificial capping - report true values for analysis
    # Values outside [0, 1] are meaningful:
    # - > 1.0 indicates overcorrection
    # - < 0.0 indicates inverse effect
    return recovery


def compute_recovery_clamped(
    patched: float,
    corrupted: float,
    clean: float,
    min_val: float = -1.0,
    max_val: float = 2.0,
) -> float:
    """
    Compute recovery with clamping (legacy behavior).
    
    Use compute_recovery() for accurate measurements.
    This function is provided for backwards compatibility.
    
    Args:
        patched: Output after patching
        corrupted: Corrupted/baseline output
        clean: Clean/target output
        min_val: Minimum clamp value
        max_val: Maximum clamp value
        
    Returns:
        Recovery score clamped to [min_val, max_val]
    """
    gap = clean - corrupted
    if abs(gap) < 1e-6:
        return 1.0  # Legacy behavior
    recovery = (patched - corrupted) / gap
    return max(min_val, min(max_val, recovery))


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(
    results: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """Save results to JSON."""
    import json
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to lists
    def convert(obj):
        if isinstance(obj, Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load results from JSON."""
    import json
    
    with open(path, "r") as f:
        return json.load(f)
