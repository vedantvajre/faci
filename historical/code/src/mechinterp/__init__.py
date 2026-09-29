"""
MechInterp: Open-Source Mechanistic Interpretability for Transformers

A unified framework for understanding transformer models through:
- Tier 1: Feature Direction Analysis
- Tier 2: Linear Probing
- Tier 3: Activation Patching / Circuit Analysis

Supports encoder, decoder, and encoder-decoder architectures.

Example usage:
    from mechinterp import MechanisticAnalyzer
    
    # Analyze a model
    analyzer = MechanisticAnalyzer("bert-base-uncased")
    results = analyzer.analyze(
        positive_texts=["I love this movie", "Great film!"],
        negative_texts=["I hate this movie", "Terrible film!"],
        task="sentiment",
    )
    
    # Access results
    print(f"Tier 1 separation: {results.tier1.separation}")
    print(f"Tier 2 best layer: {results.tier2.best_layer}")
    print(f"Tier 3 circuit size: {results.tier3.circuit_size}")
"""

__version__ = "0.1.0"

# Core imports
from mechinterp.core import (
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
    # Registries
    register_model,
    register_task,
    get_model,
    get_task,
    list_models,
    list_tasks,
    # Config
    load_config,
    load_experiment_config,
    # Utils
    set_seed,
    get_device,
)

# Tier analyzers
from mechinterp.tiers import (
    Tier1Analyzer,
    Tier2Analyzer,
    Tier3Analyzer,
)

# Import architectures to register models
from mechinterp import architectures
from mechinterp import tasks
from mechinterp import baselines


class MechanisticAnalyzer:
    """
    Unified interface for mechanistic interpretability analysis.
    
    Automatically detects architecture and runs appropriate analyses.
    """

    def __init__(
        self,
        model_name: str,
        device: str = None,
        task: str = "features",
        **kwargs
    ):
        """
        Initialize analyzer.
        
        Args:
            model_name: HuggingFace model name or registered name
            device: Device to use (auto-detected if None)
            task: Model task type (features, classification, lm)
        """
        self.model_name = model_name
        self.device = device or str(get_device())
        
        # Load model
        if model_name in list_models():
            self.model = get_model(model_name, device=self.device, task=task, **kwargs)
        else:
            # Try to create wrapper based on model type
            self.model = self._create_wrapper(model_name, task, **kwargs)
        
        self.model.load()
        
        # Initialize tier analyzers
        self.tier1 = Tier1Analyzer(self.model)
        self.tier2 = Tier2Analyzer(self.model)
        self.tier3 = Tier3Analyzer(self.model)

    def _create_wrapper(self, model_name: str, task: str, **kwargs):
        """Create appropriate wrapper for model."""
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(model_name)
        
        # Detect architecture
        if hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder:
            from mechinterp.architectures import create_enc_dec_wrapper
            return create_enc_dec_wrapper(model_name, self.device, **kwargs)
        elif hasattr(config, "is_decoder") and config.is_decoder:
            from mechinterp.architectures import create_decoder_wrapper
            return create_decoder_wrapper(model_name, self.device, task=task, **kwargs)
        else:
            from mechinterp.architectures import create_encoder_wrapper
            return create_encoder_wrapper(model_name, self.device, task=task, **kwargs)

    @property
    def architecture(self) -> Architecture:
        """Get model architecture type."""
        return self.model.architecture

    @property
    def num_layers(self) -> int:
        """Get number of layers."""
        return self.model.num_layers

    @property
    def num_heads(self) -> int:
        """Get number of attention heads."""
        return self.model.num_heads

    def analyze(
        self,
        positive_texts: list,
        negative_texts: list,
        tiers: list = None,
        **kwargs
    ) -> AnalysisResult:
        """
        Run full mechanistic analysis.
        
        Args:
            positive_texts: Examples with target feature
            negative_texts: Examples without target feature
            tiers: Which tiers to run (default: all)
            
        Returns:
            AnalysisResult with all tier results
        """
        if tiers is None:
            tiers = [1, 2, 3]
        
        tier1_result = None
        tier2_result = None
        tier3_result = None
        
        if 1 in tiers:
            tier1_result = self.tier1.analyze(positive_texts, negative_texts, **kwargs)
            # Pass feature direction to tier3 for better hidden state projection
            if tier1_result and tier1_result.direction is not None:
                self.tier3.set_feature_direction(tier1_result.direction)
        
        if 2 in tiers:
            tier2_result = self.tier2.analyze(positive_texts, negative_texts, **kwargs)
        
        if 3 in tiers:
            # If we didn't run tier1 but need tier3, compute direction first
            if tier1_result is None and not hasattr(self.tier3, '_feature_direction'):
                # Quick direction computation for tier3
                direction = self.tier1.compute_feature_direction(
                    positive_texts, negative_texts
                )
                self.tier3.set_feature_direction(direction)
            tier3_result = self.tier3.analyze(positive_texts, negative_texts, **kwargs)
        
        return AnalysisResult(
            model_name=self.model_name,
            task_name=kwargs.get("task", "unknown"),
            architecture=self.architecture,
            tier1=tier1_result,
            tier2=tier2_result,
            tier3=tier3_result,
        )

    def compute_direction(
        self,
        positive_texts: list,
        negative_texts: list,
        layer: int = -1,
        **kwargs
    ):
        """Compute feature direction (Tier 1)."""
        self.tier1.layer = layer
        return self.tier1.compute_feature_direction(
            positive_texts, negative_texts, **kwargs
        )

    def probe_layers(
        self,
        texts: list,
        labels: list,
        **kwargs
    ):
        """Probe all layers (Tier 2)."""
        return self.tier2.probe_all_layers(texts, labels, **kwargs)

    def find_circuit(
        self,
        clean_text: str,
        corrupted_text: str,
        **kwargs
    ):
        """Find important circuit components (Tier 3)."""
        return self.tier3.find_important_layers(
            clean_text, corrupted_text, **kwargs
        )

    def analyze_with_multiple_examples(
        self,
        positive_texts: list,
        negative_texts: list,
        num_pairs: int = 10,
        threshold: float = 0.5,
        **kwargs
    ) -> Tier3Result:
        """
        Run Tier 3 analysis with multiple example pairs for robust estimates.
        
        Averages recovery scores across multiple (positive, negative) pairs
        to reduce variance from single-example noise.
        
        Args:
            positive_texts: Examples with target feature/behavior
            negative_texts: Examples without target feature/behavior
            num_pairs: Number of example pairs to average over
            threshold: Recovery threshold for circuit membership (0-1)
            
        Returns:
            Tier3Result with averaged recovery scores
        """
        # Ensure tier3 has a feature direction for meaningful output projection
        if not hasattr(self.tier3, '_feature_direction') or self.tier3._feature_direction is None:
            direction = self.tier1.compute_feature_direction(
                positive_texts, negative_texts
            )
            self.tier3.set_feature_direction(direction)
        
        return self.tier3.analyze_with_multiple_examples(
            positive_texts,
            negative_texts,
            num_pairs=num_pairs,
            threshold=threshold,
            **kwargs
        )

    def get_recovery_curve(
        self,
        positive_texts: list,
        negative_texts: list,
        num_pairs: int = 5,
        **kwargs
    ):
        """
        Get detailed recovery curve for visualization.
        
        Returns layer-by-layer recovery statistics including mean, std,
        and circuit sizes at multiple thresholds.
        """
        return self.tier3.get_recovery_curve(
            positive_texts,
            negative_texts,
            num_pairs=num_pairs,
            **kwargs
        )


# Convenience function
def analyze(
    model_name: str,
    positive_texts: list,
    negative_texts: list,
    **kwargs
) -> AnalysisResult:
    """
    Quick analysis function.
    
    Args:
        model_name: Model to analyze
        positive_texts: Examples with feature
        negative_texts: Examples without feature
        
    Returns:
        AnalysisResult
    """
    analyzer = MechanisticAnalyzer(model_name, **kwargs)
    return analyzer.analyze(positive_texts, negative_texts, **kwargs)


__all__ = [
    # Main class
    "MechanisticAnalyzer",
    # Quick function
    "analyze",
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
    # Tier analyzers
    "Tier1Analyzer",
    "Tier2Analyzer",
    "Tier3Analyzer",
    # Registries
    "register_model",
    "register_task",
    "get_model",
    "get_task",
    "list_models",
    "list_tasks",
    # Utils
    "set_seed",
    "get_device",
    # Version
    "__version__",
]
