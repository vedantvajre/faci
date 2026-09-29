"""
Base architecture wrapper with hook management.
"""

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

from mechinterp.core.base import (
    BaseModelWrapper,
    Architecture,
    ExtractionPosition,
    ActivationCache,
)


class HookedModelWrapper(BaseModelWrapper):
    """
    Base class for HuggingFace model wrappers with hook management.
    
    Provides common functionality for:
    - Model/tokenizer loading
    - Forward hooks for activation capture
    - Activation patching utilities
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        model_class: Optional[type] = None,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self._model_class = model_class or AutoModel
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._cache = ActivationCache()
        self._patch_spec: Dict[str, Tensor] = {}

    def load(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = self._model_class.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    @property
    def model(self) -> nn.Module:
        """Get the underlying model."""
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def _tokenize(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Tensor]:
        """Tokenize input text."""
        if isinstance(text, str):
            text = [text]
        
        defaults = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 512,
        }
        defaults.update(kwargs)
        
        inputs = self.tokenizer(text, **defaults)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _clear_cache(self) -> None:
        """Clear the activation cache."""
        self._cache = ActivationCache()

    def _make_hook(
        self,
        name: str,
        component: str = "hidden",
    ) -> Callable:
        """
        Create a forward hook that caches activations.
        
        Args:
            name: Name to store activation under
            component: Type of component (hidden, attention, mlp)
        """
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            # Check if we should patch
            if name in self._patch_spec:
                activation = self._patch_spec[name].to(activation.device)
                if isinstance(output, tuple):
                    output = (activation,) + output[1:]
                else:
                    output = activation
            
            # Cache the activation
            if component == "hidden":
                self._cache.hidden_states[name] = activation.detach()
            elif component == "attention":
                self._cache.attentions[name] = activation.detach()
            elif component == "mlp":
                self._cache.mlp_outputs[name] = activation.detach()
            
            return output
        
        return hook

    def _make_patch_hook(
        self,
        name: str,
        patch_value: Tensor,
    ) -> Callable:
        """Create a hook that patches activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (patch_value.to(output[0].device),) + output[1:]
            return patch_value.to(output.device)
        return hook

    @abstractmethod
    def _get_layer_modules(self) -> List[nn.Module]:
        """Get the list of layer modules for hooking."""
        pass

    @abstractmethod
    def _get_attention_modules(self) -> List[nn.Module]:
        """Get the list of attention modules for hooking."""
        pass

    @abstractmethod
    def _get_mlp_modules(self) -> List[nn.Module]:
        """Get the list of MLP modules for hooking."""
        pass

    def _register_hooks(self, cache_hidden: bool = True) -> None:
        """Register hooks on all layers."""
        self._clear_hooks()
        
        if cache_hidden:
            for i, module in enumerate(self._get_layer_modules()):
                hook = module.register_forward_hook(
                    self._make_hook(f"layer_{i}", "hidden")
                )
                self._hooks.append(hook)

    def _extract_at_position(
        self,
        hidden_states: Tensor,
        position: ExtractionPosition,
        attention_mask: Optional[Tensor] = None,
        n: int = 4,
    ) -> Tensor:
        """
        Extract representations at specified position.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            position: Where to extract
            attention_mask: (batch, seq_len) optional mask
            n: Number of tokens for LAST_N_MEAN
            
        Returns:
            Tensor of shape (batch, hidden_size)
        """
        if position == ExtractionPosition.CLS:
            return hidden_states[:, 0, :]
        
        elif position == ExtractionPosition.FINAL_TOKEN:
            if attention_mask is not None:
                # Get last non-padding position for each sequence
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                return hidden_states[batch_indices, seq_lengths.long(), :]
            else:
                return hidden_states[:, -1, :]
        
        elif position == ExtractionPosition.MEAN_ALL:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                return hidden_states.mean(dim=1)
        
        elif position == ExtractionPosition.LAST_N_MEAN:
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1)
                results = []
                for i, length in enumerate(seq_lengths):
                    start = max(0, length - n)
                    results.append(hidden_states[i, start:length, :].mean(dim=0))
                return torch.stack(results)
            else:
                return hidden_states[:, -n:, :].mean(dim=1)
        
        else:
            raise ValueError(f"Unsupported extraction position: {position}")

    def forward(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Tuple[Tensor, ActivationCache]:
        """Run forward pass and return output + cached activations."""
        self._clear_cache()
        self._register_hooks()
        
        inputs = self._tokenize(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        self._clear_hooks()
        
        # Get logits or last hidden state
        if hasattr(outputs, "logits"):
            output = outputs.logits
        else:
            output = outputs.last_hidden_state
        
        return output, self._cache

    def get_embedding(
        self,
        text: Union[str, List[str]],
        layer: int = -1,
        position: ExtractionPosition = ExtractionPosition.CLS,
    ) -> Tensor:
        """Get embeddings at specified layer and position."""
        inputs = self._tokenize(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get hidden states at specified layer
        hidden_states = outputs.hidden_states[layer]
        
        # Extract at position
        return self._extract_at_position(
            hidden_states,
            position,
            attention_mask=inputs.get("attention_mask"),
        )

    def run_with_patch(
        self,
        text: Union[str, List[str]],
        patch_spec: Dict[str, Tensor],
    ) -> Tensor:
        """Run forward pass with activations patched."""
        self._clear_cache()
        self._patch_spec = patch_spec
        self._register_hooks()
        
        inputs = self._tokenize(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        self._clear_hooks()
        self._patch_spec = {}
        
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs.last_hidden_state
