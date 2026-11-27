"""
Decoder architecture wrappers (GPT-2, GPT-Neo, etc.).
"""

from typing import List, Optional, Union

import torch
from torch import Tensor, nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
)

from mechinterp.core.base import Architecture, ExtractionPosition
from mechinterp.core.registry import register_model
from mechinterp.architectures.base import HookedModelWrapper


class DecoderWrapper(HookedModelWrapper):
    """
    Wrapper for decoder-only models (GPT-2, GPT-Neo, etc.).
    
    Supports:
    - Causal language modeling
    - Sequence classification (with pad token)
    - Feature extraction at any layer
    
    Extraction positions:
    - FINAL_TOKEN: Last token in sequence (default)
    - MEAN_ALL: Mean of all tokens
    - LAST_N_MEAN: Mean of last N tokens
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        model_class: Optional[type] = None,
        default_position: ExtractionPosition = ExtractionPosition.FINAL_TOKEN,
        **kwargs
    ):
        super().__init__(model_name, device, model_class, **kwargs)
        self._default_position = default_position

    @property
    def architecture(self) -> Architecture:
        return Architecture.DECODER

    @property
    def num_layers(self) -> int:
        config = self.model.config
        return config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        config = self.model.config
        return config.n_head if hasattr(config, "n_head") else config.num_attention_heads

    @property
    def hidden_size(self) -> int:
        config = self.model.config
        return config.n_embd if hasattr(config, "n_embd") else config.hidden_size

    @property
    def default_position(self) -> ExtractionPosition:
        """Default extraction position for decoders is final token."""
        return self._default_position

    def load(self) -> None:
        """Load model and ensure pad token exists."""
        super().load()
        # GPT-2 doesn't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if hasattr(self.model, "config"):
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _get_transformer(self) -> nn.Module:
        """Get the transformer module."""
        model = self.model
        if hasattr(model, "transformer"):
            return model.transformer
        elif hasattr(model, "gpt2"):
            return model.gpt2
        elif hasattr(model, "gpt_neox"):
            return model.gpt_neox
        elif hasattr(model, "model"):
            return model.model
        elif hasattr(model, "h"):
            # GPT2Model itself is the transformer
            return model
        else:
            raise ValueError(f"Cannot find transformer in {type(model)}")

    def _get_layer_modules(self) -> List[nn.Module]:
        """Get the list of transformer layers."""
        transformer = self._get_transformer()
        if hasattr(transformer, "h"):
            return list(transformer.h)
        elif hasattr(transformer, "layers"):
            return list(transformer.layers)
        else:
            raise ValueError("Cannot find layers in transformer")

    def _get_attention_modules(self) -> List[nn.Module]:
        """Get the list of attention modules."""
        layers = self._get_layer_modules()
        return [layer.attn for layer in layers]

    def _get_mlp_modules(self) -> List[nn.Module]:
        """Get the list of MLP modules."""
        layers = self._get_layer_modules()
        return [layer.mlp for layer in layers]

    def get_embedding(
        self,
        text: Union[str, List[str]],
        layer: int = -1,
        position: Optional[ExtractionPosition] = None,
    ) -> Tensor:
        """
        Get embeddings at specified layer and position.
        
        Args:
            text: Input text(s)
            layer: Layer index (-1 for final)
            position: Where to extract (default: final_token)
        """
        if position is None:
            position = self.default_position
        
        inputs = self._tokenize(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[layer]
        
        return self._extract_at_position(
            hidden_states,
            position,
            attention_mask=inputs.get("attention_mask"),
        )

    def generate_next_token_probs(
        self,
        text: Union[str, List[str]],
    ) -> Tensor:
        """
        Get probability distribution over next token.
        
        Returns:
            (batch, vocab_size) tensor of probabilities
        """
        inputs = self._tokenize(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits for last position
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        return probs


@register_model("gpt2")
class GPT2SmallWrapper(DecoderWrapper):
    """GPT-2 Small (124M) wrapper."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        task: str = "lm",
        position: ExtractionPosition = ExtractionPosition.FINAL_TOKEN,
        **kwargs
    ):
        model_class = {
            "lm": AutoModelForCausalLM,
            "classification": AutoModelForSequenceClassification,
            "features": AutoModel,
        }.get(task, AutoModelForCausalLM)
        
        super().__init__(
            model_name="gpt2",
            device=device,
            model_class=model_class,
            default_position=position,
            **kwargs
        )


@register_model("gpt2-medium")
class GPT2MediumWrapper(DecoderWrapper):
    """GPT-2 Medium (355M) wrapper."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        task: str = "lm",
        position: ExtractionPosition = ExtractionPosition.FINAL_TOKEN,
        **kwargs
    ):
        model_class = {
            "lm": AutoModelForCausalLM,
            "classification": AutoModelForSequenceClassification,
            "features": AutoModel,
        }.get(task, AutoModelForCausalLM)
        
        super().__init__(
            model_name="gpt2-medium",
            device=device,
            model_class=model_class,
            default_position=position,
            **kwargs
        )


@register_model("gpt2-large")
class GPT2LargeWrapper(DecoderWrapper):
    """GPT-2 Large (774M) wrapper."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        task: str = "lm",
        position: ExtractionPosition = ExtractionPosition.FINAL_TOKEN,
        **kwargs
    ):
        model_class = {
            "lm": AutoModelForCausalLM,
            "classification": AutoModelForSequenceClassification,
            "features": AutoModel,
        }.get(task, AutoModelForCausalLM)
        
        super().__init__(
            model_name="gpt2-large",
            device=device,
            model_class=model_class,
            default_position=position,
            **kwargs
        )


# Factory function
def create_decoder_wrapper(
    model_name: str,
    device: Optional[str] = None,
    task: str = "lm",
    position: ExtractionPosition = ExtractionPosition.FINAL_TOKEN,
    **kwargs
) -> DecoderWrapper:
    """
    Create a decoder wrapper for any HuggingFace decoder model.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        task: Task type (lm, classification, features)
        position: Default extraction position
        
    Returns:
        DecoderWrapper instance
    """
    model_class = {
        "lm": AutoModelForCausalLM,
        "classification": AutoModelForSequenceClassification,
        "features": AutoModel,
    }.get(task, AutoModelForCausalLM)
    
    wrapper = DecoderWrapper(
        model_name=model_name,
        device=device,
        model_class=model_class,
        default_position=position,
        **kwargs
    )
    return wrapper
