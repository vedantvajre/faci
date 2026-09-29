"""
Abstract base classes for all components.

These define the interface that all implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


# =============================================================================
# Enums
# =============================================================================


class Architecture(Enum):
    """Transformer architecture types."""
    ENCODER = "encoder"
    DECODER = "decoder"
    ENCODER_DECODER = "encoder_decoder"


class ExtractionPosition(Enum):
    """Where to extract representations."""
    CLS = "cls"                    # [CLS] token (encoder)
    FINAL_TOKEN = "final_token"   # Last token (decoder)
    MEAN_ALL = "mean_all"         # Mean of all tokens
    LAST_N_MEAN = "last_n_mean"   # Mean of last N tokens
    ENCODER_FINAL = "encoder_final"       # Encoder output (enc-dec)
    DECODER_FINAL = "decoder_final"       # Decoder output (enc-dec)
    CROSS_ATTN = "cross_attn"             # Cross-attention (enc-dec)


class PatchTarget(Enum):
    """What to patch in Tier 3."""
    HEAD = "head"                 # Individual attention head
    LAYER = "layer"               # Entire layer
    MLP = "mlp"                   # MLP block only
    ATTN = "attn"                 # Attention block only
    ENCODER_ONLY = "encoder_only"         # Encoder layers (enc-dec)
    DECODER_ONLY = "decoder_only"         # Decoder layers (enc-dec)
    CROSS_ATTN_ONLY = "cross_attn_only"   # Cross-attention (enc-dec)
    ALL = "all"                   # All components


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ActivationCache:
    """Stores activations from a forward pass."""
    hidden_states: Dict[str, Tensor] = field(default_factory=dict)
    attentions: Dict[str, Tensor] = field(default_factory=dict)
    mlp_outputs: Dict[str, Tensor] = field(default_factory=dict)

    def get_layer(self, layer: int, component: str = "hidden") -> Optional[Tensor]:
        """Get activation for a specific layer."""
        key = f"layer_{layer}"
        if component == "hidden":
            return self.hidden_states.get(key)
        elif component == "attention":
            return self.attentions.get(key)
        elif component == "mlp":
            return self.mlp_outputs.get(key)
        return None


@dataclass
class Tier1Result:
    """Results from Tier 1 analysis (feature directions)."""
    direction: Tensor                     # Feature direction vector
    layer: int                            # Layer where computed
    positive_projections: List[float]     # Projections of positive examples
    negative_projections: List[float]     # Projections of negative examples
    separation: float                     # Separation metric (e.g., AUC)
    position: ExtractionPosition          # Where extracted


@dataclass
class Tier2Result:
    """Results from Tier 2 analysis (linear probing)."""
    layer_accuracies: Dict[int, float]    # Accuracy per layer
    emergence_layer: int                  # First layer > threshold
    best_layer: int                       # Layer with highest accuracy
    best_accuracy: float                  # Highest accuracy
    probes: Dict[int, Any]                # Trained probe per layer (optional)


@dataclass 
class Tier3Result:
    """Results from Tier 3 analysis (activation patching)."""
    component_effects: Dict[str, float]   # Effect per component
    important_components: List[str]       # Components with effect > threshold
    circuit_size: int                     # Number of important components
    total_components: int                 # Total components tested
    sparsity: float                       # Fraction of unimportant components


@dataclass
class AnalysisResult:
    """Combined results from full analysis."""
    model_name: str
    task_name: str
    architecture: Architecture
    tier1: Optional[Tier1Result] = None
    tier2: Optional[Tier2Result] = None
    tier3: Optional[Tier3Result] = None
    baselines: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Abstract Base Classes
# =============================================================================


class BaseModelWrapper(ABC):
    """
    Abstract base class for model wrappers.
    
    Wraps HuggingFace models with hooks for activation extraction.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._hooks = []

    @property
    @abstractmethod
    def architecture(self) -> Architecture:
        """Return the architecture type."""
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of layers."""
        pass

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Return the number of attention heads per layer."""
        pass

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the hidden dimension."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def forward(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Tuple[Tensor, ActivationCache]:
        """
        Run forward pass and return output + cached activations.
        
        Returns:
            output: Model output (logits or hidden states)
            cache: Cached activations from all layers
        """
        pass

    @abstractmethod
    def get_embedding(
        self,
        text: Union[str, List[str]],
        layer: int = -1,
        position: ExtractionPosition = ExtractionPosition.CLS,
    ) -> Tensor:
        """
        Get embeddings at specified layer and position.
        
        Args:
            text: Input text(s)
            layer: Layer index (-1 for final)
            position: Where to extract (CLS, final token, etc.)
            
        Returns:
            Tensor of shape (batch, hidden_size)
        """
        pass

    @abstractmethod
    def run_with_patch(
        self,
        text: Union[str, List[str]],
        patch_spec: Dict[str, Tensor],
    ) -> Tensor:
        """
        Run forward pass with activations patched.
        
        Args:
            text: Input text(s)
            patch_spec: Dict mapping component names to replacement tensors
            
        Returns:
            Model output with patched activations
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model_name}, arch={self.architecture.value})"


class BaseTask(ABC):
    """
    Abstract base class for tasks.
    
    Handles data loading, formatting, and evaluation for each architecture.
    """

    def __init__(self, **kwargs):
        self._data = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Task name."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes (for classification)."""
        pass

    @abstractmethod
    def load_data(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> None:
        """Load task data."""
        pass

    @abstractmethod
    def get_positive_examples(self, n: Optional[int] = None) -> List[str]:
        """Get positive examples (class 1)."""
        pass

    @abstractmethod
    def get_negative_examples(self, n: Optional[int] = None) -> List[str]:
        """Get negative examples (class 0)."""
        pass

    @abstractmethod
    def format_input(
        self,
        example: Dict[str, Any],
        architecture: Architecture,
    ) -> str:
        """
        Format input for the given architecture.
        
        Args:
            example: Raw example from dataset
            architecture: Target architecture
            
        Returns:
            Formatted input string
        """
        pass

    @abstractmethod
    def get_label(self, example: Dict[str, Any]) -> int:
        """Get the label for an example."""
        pass

    @abstractmethod
    def evaluate(
        self,
        predictions: List[Any],
        labels: List[int],
    ) -> Dict[str, float]:
        """Evaluate predictions against labels."""
        pass


class BaseTierAnalyzer(ABC):
    """
    Abstract base class for tier analyzers.
    """

    def __init__(self, model: BaseModelWrapper, **kwargs):
        self.model = model

    @property
    @abstractmethod
    def tier(self) -> int:
        """Tier number (1, 2, or 3)."""
        pass

    @abstractmethod
    def analyze(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        **kwargs
    ) -> Union[Tier1Result, Tier2Result, Tier3Result]:
        """Run the tier analysis."""
        pass


class BaselineMethod(ABC):
    """
    Abstract base class for baseline methods.
    """

    def __init__(self, model: BaseModelWrapper, **kwargs):
        self.model = model

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name."""
        pass

    @property
    @abstractmethod
    def is_causal(self) -> bool:
        """Whether this method provides causal evidence."""
        pass

    @abstractmethod
    def explain(
        self,
        text: str,
        target: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single input.
        
        Returns dict with method-specific explanation data.
        """
        pass

    @abstractmethod
    def explain_batch(
        self,
        texts: List[str],
        targets: Optional[List[int]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate explanations for multiple inputs."""
        pass
