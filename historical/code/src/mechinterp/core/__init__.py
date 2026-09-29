"""
Core module for mechinterp.

Provides base classes, registries, and utilities.
"""

from mechinterp.core.base import (
    # Enums
    Architecture,
    ExtractionPosition,
    PatchTarget,
    # Data classes
    ActivationCache,
    Tier1Result,
    Tier2Result,
    Tier3Result,
    AnalysisResult,
    # Base classes
    BaseModelWrapper,
    BaseTask,
    BaseTierAnalyzer,
    BaselineMethod,
)

from mechinterp.core.registry import (
    Registry,
    MODEL_REGISTRY,
    TASK_REGISTRY,
    TIER_REGISTRY,
    BASELINE_REGISTRY,
    register_model,
    register_task,
    register_tier,
    register_baseline,
    get_model,
    get_task,
    get_tier,
    get_baseline,
    list_models,
    list_tasks,
    list_tiers,
    list_baselines,
)

from mechinterp.core.config import (
    ModelConfig,
    TaskConfig,
    ExperimentConfig,
    load_yaml,
    load_config,
    load_model_config,
    load_task_config,
    load_experiment_config,
    merge_configs,
    save_config,
)

from mechinterp.core.utils import (
    setup_logging,
    set_seed,
    get_device,
    cosine_similarity,
    compute_direction,
    project_onto_direction,
    compute_logit_diff,
    compute_recovery,
    ensure_dir,
    save_results,
    load_results,
)

__all__ = [
    # Enums
    "Architecture",
    "ExtractionPosition",
    "PatchTarget",
    # Data classes
    "ActivationCache",
    "Tier1Result",
    "Tier2Result",
    "Tier3Result",
    "AnalysisResult",
    # Base classes
    "BaseModelWrapper",
    "BaseTask",
    "BaseTierAnalyzer",
    "BaselineMethod",
    # Registry
    "Registry",
    "MODEL_REGISTRY",
    "TASK_REGISTRY",
    "TIER_REGISTRY",
    "BASELINE_REGISTRY",
    "register_model",
    "register_task",
    "register_tier",
    "register_baseline",
    "get_model",
    "get_task",
    "get_tier",
    "get_baseline",
    "list_models",
    "list_tasks",
    "list_tiers",
    "list_baselines",
    # Config
    "ModelConfig",
    "TaskConfig",
    "ExperimentConfig",
    "load_yaml",
    "load_config",
    "load_model_config",
    "load_task_config",
    "load_experiment_config",
    "merge_configs",
    "save_config",
    # Utils
    "setup_logging",
    "set_seed",
    "get_device",
    "cosine_similarity",
    "compute_direction",
    "project_onto_direction",
    "compute_logit_diff",
    "compute_recovery",
    "ensure_dir",
    "save_results",
    "load_results",
]
