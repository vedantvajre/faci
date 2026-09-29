"""
Configuration loading and management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from omegaconf import OmegaConf, DictConfig


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    hf_name: str                          # HuggingFace model name
    architecture: str                      # encoder, decoder, encoder_decoder
    num_layers: int
    num_heads: int
    hidden_size: int
    extraction_positions: List[str] = field(default_factory=list)
    patch_targets: List[str] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a task."""
    name: str
    dataset: str                          # HuggingFace dataset name
    dataset_config: Optional[str] = None  # Dataset configuration
    positive_label: int = 1
    negative_label: int = 0
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 512
    templates: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    models: List[str]
    tasks: List[str]
    tiers: List[int] = field(default_factory=lambda: [1, 2, 3])
    baselines: List[str] = field(default_factory=list)
    num_positive: int = 100
    num_negative: int = 100
    num_pairs: int = 10  # Number of example pairs for Tier 3 averaging
    output_dir: str = "results"
    seed: int = 42
    device: Optional[str] = None


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(path: Union[str, Path]) -> DictConfig:
    """Load a config file as OmegaConf."""
    data = load_yaml(path)
    return OmegaConf.create(data)


def load_model_config(name: str, config_dir: Optional[Path] = None) -> ModelConfig:
    """Load a model configuration by name."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "models"
    
    path = config_dir / f"{name}.yaml"
    data = load_yaml(path)
    return ModelConfig(**data)


def load_task_config(name: str, config_dir: Optional[Path] = None) -> TaskConfig:
    """Load a task configuration by name."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "tasks"
    
    path = config_dir / f"{name}.yaml"
    data = load_yaml(path)
    return TaskConfig(**data)


def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load an experiment configuration."""
    data = load_yaml(path)
    return ExperimentConfig(**data)


def merge_configs(*configs: Dict[str, Any]) -> DictConfig:
    """Merge multiple configs, later ones override earlier."""
    merged = OmegaConf.create({})
    for cfg in configs:
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        merged = OmegaConf.merge(merged, cfg)
    return merged


def save_config(config: Union[Dict, DictConfig], path: Union[str, Path]) -> None:
    """Save a config to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config)
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
