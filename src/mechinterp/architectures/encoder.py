"""
Encoder architecture wrappers (BERT, RoBERTa, DeBERTa).
"""

from typing import List, Optional

from torch import nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)

from mechinterp.core.base import Architecture, ExtractionPosition
from mechinterp.core.registry import register_model
from mechinterp.architectures.base import HookedModelWrapper


class EncoderWrapper(HookedModelWrapper):
    """
    Wrapper for encoder-only models (BERT, RoBERTa, DeBERTa).
    
    Supports:
    - Sequence classification (CLS token)
    - Masked language modeling
    - Feature extraction at any layer
    """

    @property
    def architecture(self) -> Architecture:
        return Architecture.ENCODER

    @property
    def num_layers(self) -> int:
        config = self.model.config
        return config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        config = self.model.config
        return config.num_attention_heads

    @property
    def hidden_size(self) -> int:
        config = self.model.config
        return config.hidden_size

    @property
    def default_position(self) -> ExtractionPosition:
        """Default extraction position for encoders is CLS."""
        return ExtractionPosition.CLS

    def _get_encoder(self) -> nn.Module:
        """Get the encoder module (handles different model types)."""
        model = self.model
        # Handle different wrapper types
        if hasattr(model, "bert"):
            return model.bert.encoder
        elif hasattr(model, "roberta"):
            return model.roberta.encoder
        elif hasattr(model, "deberta"):
            return model.deberta.encoder
        elif hasattr(model, "distilbert"):
            return model.distilbert.transformer
        elif hasattr(model, "transformer"):
            # DistilBERT base model
            return model.transformer
        elif hasattr(model, "encoder"):
            return model.encoder
        else:
            raise ValueError(f"Cannot find encoder in {type(model)}")

    def _get_layer_modules(self) -> List[nn.Module]:
        """Get the list of transformer layers."""
        encoder = self._get_encoder()
        # Handle different attribute names
        if hasattr(encoder, "layer"):
            return list(encoder.layer)
        elif hasattr(encoder, "layers"):
            return list(encoder.layers)
        else:
            raise ValueError(f"Cannot find layers in encoder: {type(encoder)}")

    def _get_attention_modules(self) -> List[nn.Module]:
        """Get the list of attention modules."""
        layers = self._get_layer_modules()
        attns = []
        for layer in layers:
            if hasattr(layer, "attention"):
                attns.append(layer.attention)
            elif hasattr(layer, "self_attention"):
                attns.append(layer.self_attention)
            elif hasattr(layer, "attn"):
                attns.append(layer.attn)
            else:
                attns.append(layer)  # Fallback
        return attns

    def _get_mlp_modules(self) -> List[nn.Module]:
        """Get the list of MLP/FFN modules."""
        layers = self._get_layer_modules()
        # Different models have different names
        mlps = []
        for layer in layers:
            if hasattr(layer, "intermediate"):
                mlps.append(layer.intermediate)
            elif hasattr(layer, "ffn"):
                mlps.append(layer.ffn)
            elif hasattr(layer, "output"):
                mlps.append(layer.output)
            else:
                mlps.append(layer)  # Fallback
        return mlps


@register_model("bert-base-uncased")
class BertBaseWrapper(EncoderWrapper):
    """BERT Base Uncased wrapper."""
    
    def __init__(self, device: Optional[str] = None, task: str = "classification", **kwargs):
        model_class = {
            "classification": AutoModelForSequenceClassification,
            "mlm": AutoModelForMaskedLM,
            "features": AutoModel,
        }.get(task, AutoModel)
        
        super().__init__(
            model_name="bert-base-uncased",
            device=device,
            model_class=model_class,
            **kwargs
        )


@register_model("bert-large-uncased")
class BertLargeWrapper(EncoderWrapper):
    """BERT Large Uncased wrapper."""
    
    def __init__(self, device: Optional[str] = None, task: str = "classification", **kwargs):
        model_class = {
            "classification": AutoModelForSequenceClassification,
            "mlm": AutoModelForMaskedLM,
            "features": AutoModel,
        }.get(task, AutoModel)
        
        super().__init__(
            model_name="bert-large-uncased",
            device=device,
            model_class=model_class,
            **kwargs
        )


@register_model("roberta-base")
class RobertaBaseWrapper(EncoderWrapper):
    """RoBERTa Base wrapper."""
    
    def __init__(self, device: Optional[str] = None, task: str = "classification", **kwargs):
        model_class = {
            "classification": AutoModelForSequenceClassification,
            "mlm": AutoModelForMaskedLM,
            "features": AutoModel,
        }.get(task, AutoModel)
        
        super().__init__(
            model_name="roberta-base",
            device=device,
            model_class=model_class,
            **kwargs
        )


@register_model("roberta-large")
class RobertaLargeWrapper(EncoderWrapper):
    """RoBERTa Large wrapper."""
    
    def __init__(self, device: Optional[str] = None, task: str = "classification", **kwargs):
        model_class = {
            "classification": AutoModelForSequenceClassification,
            "mlm": AutoModelForMaskedLM,
            "features": AutoModel,
        }.get(task, AutoModel)
        
        super().__init__(
            model_name="roberta-large",
            device=device,
            model_class=model_class,
            **kwargs
        )


@register_model("distilbert-base-uncased")
class DistilBertBaseWrapper(EncoderWrapper):
    """DistilBERT Base Uncased wrapper."""
    
    def __init__(self, device: Optional[str] = None, task: str = "classification", **kwargs):
        model_class = {
            "classification": AutoModelForSequenceClassification,
            "mlm": AutoModelForMaskedLM,
            "features": AutoModel,
        }.get(task, AutoModel)
        
        super().__init__(
            model_name="distilbert-base-uncased",
            device=device,
            model_class=model_class,
            **kwargs
        )
    
    @property
    def num_layers(self) -> int:
        config = self.model.config
        return config.n_layers if hasattr(config, "n_layers") else config.num_hidden_layers


# Factory function for custom encoder models
def create_encoder_wrapper(
    model_name: str,
    device: Optional[str] = None,
    task: str = "classification",
    **kwargs
) -> EncoderWrapper:
    """
    Create an encoder wrapper for any HuggingFace encoder model.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        task: Task type (classification, mlm, features)
        
    Returns:
        EncoderWrapper instance
    """
    model_class = {
        "classification": AutoModelForSequenceClassification,
        "mlm": AutoModelForMaskedLM,
        "features": AutoModel,
    }.get(task, AutoModel)
    
    wrapper = EncoderWrapper(
        model_name=model_name,
        device=device,
        model_class=model_class,
        **kwargs
    )
    return wrapper
