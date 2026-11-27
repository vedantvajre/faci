"""
Encoder-decoder architecture wrappers (T5, BART, mT5).
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
)

from mechinterp.core.base import (
    Architecture,
    ExtractionPosition,
    ActivationCache,
    PatchTarget,
)
from mechinterp.core.registry import register_model
from mechinterp.architectures.base import HookedModelWrapper


class EncDecActivationCache(ActivationCache):
    """Extended cache for encoder-decoder models."""
    
    def __init__(self):
        super().__init__()
        self.encoder_hidden: Dict[str, Tensor] = {}
        self.decoder_hidden: Dict[str, Tensor] = {}
        self.cross_attention: Dict[str, Tensor] = {}


class EncoderDecoderWrapper(HookedModelWrapper):
    """
    Wrapper for encoder-decoder models (T5, BART, mT5).
    
    Supports:
    - Seq2seq generation
    - Feature extraction from encoder, decoder, or cross-attention
    - Patching at encoder, decoder, or cross-attention
    
    Extraction positions:
    - ENCODER_FINAL: Final encoder hidden state
    - DECODER_FINAL: Final decoder hidden state
    - CROSS_ATTN: Cross-attention output
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        model_class: Optional[type] = None,
        default_position: ExtractionPosition = ExtractionPosition.ENCODER_FINAL,
        **kwargs
    ):
        super().__init__(model_name, device, model_class, **kwargs)
        self._default_position = default_position
        self._enc_dec_cache = EncDecActivationCache()

    @property
    def architecture(self) -> Architecture:
        return Architecture.ENCODER_DECODER

    @property
    def num_encoder_layers(self) -> int:
        """Number of encoder layers (without embedding layer)."""
        config = self.model.config
        # T5 uses num_layers for encoder
        if hasattr(config, "num_layers"):
            return config.num_layers
        # BART uses encoder_layers
        if hasattr(config, "encoder_layers"):
            return config.encoder_layers
        # Fallback to num_hidden_layers (might be total)
        if hasattr(config, "num_hidden_layers"):
            return config.num_hidden_layers
        raise ValueError(f"Cannot determine encoder layers for {type(config)}")

    @property
    def num_decoder_layers(self) -> int:
        """Number of decoder layers (without embedding layer)."""
        config = self.model.config
        # T5 uses num_decoder_layers
        if hasattr(config, "num_decoder_layers"):
            return config.num_decoder_layers
        # BART uses decoder_layers
        if hasattr(config, "decoder_layers"):
            return config.decoder_layers
        # Fallback to same as encoder
        return self.num_encoder_layers

    @property
    def num_layers(self) -> int:
        """
        Number of layers available for analysis.
        
        For encoder-decoder models, this depends on the extraction position:
        - ENCODER_FINAL: returns encoder layers only
        - DECODER_FINAL: returns decoder layers only
        - Otherwise: returns encoder layers (default for sentiment/classification tasks)
        
        Use num_total_layers for the full count.
        """
        if self._default_position == ExtractionPosition.DECODER_FINAL:
            n = self.num_decoder_layers
        else:
            # For encoder extraction (default for classification), use encoder layers
            n = self.num_encoder_layers
        return n
    
    @property
    def num_total_layers(self) -> int:
        """Total layers (encoder + decoder)."""
        return self.num_encoder_layers + self.num_decoder_layers
    
    def debug_config(self) -> dict:
        """Return debug info about model configuration."""
        config = self.model.config
        return {
            "model_name": self.model_name,
            "default_position": self._default_position.value if hasattr(self._default_position, 'value') else str(self._default_position),
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "num_layers": self.num_layers,
            "config_attrs": {
                "num_layers": getattr(config, "num_layers", None),
                "encoder_layers": getattr(config, "encoder_layers", None),
                "decoder_layers": getattr(config, "decoder_layers", None),
                "num_hidden_layers": getattr(config, "num_hidden_layers", None),
                "num_decoder_layers": getattr(config, "num_decoder_layers", None),
            }
        }

    @property
    def num_heads(self) -> int:
        config = self.model.config
        return config.num_heads if hasattr(config, "num_heads") else config.num_attention_heads

    @property
    def hidden_size(self) -> int:
        config = self.model.config
        return config.d_model if hasattr(config, "d_model") else config.hidden_size

    @property
    def default_position(self) -> ExtractionPosition:
        return self._default_position

    def _get_encoder(self) -> nn.Module:
        """Get the encoder module."""
        model = self.model
        if hasattr(model, "encoder"):
            return model.encoder
        elif hasattr(model, "model") and hasattr(model.model, "encoder"):
            return model.model.encoder
        else:
            raise ValueError(f"Cannot find encoder in {type(model)}")

    def _get_decoder(self) -> nn.Module:
        """Get the decoder module."""
        model = self.model
        if hasattr(model, "decoder"):
            return model.decoder
        elif hasattr(model, "model") and hasattr(model.model, "decoder"):
            return model.model.decoder
        else:
            raise ValueError(f"Cannot find decoder in {type(model)}")

    def _get_layer_modules(self) -> List[nn.Module]:
        """
        Get layer modules for patching.
        
        Returns encoder layers for ENCODER_FINAL position,
        decoder layers for DECODER_FINAL position.
        """
        if self._default_position == ExtractionPosition.DECODER_FINAL:
            return self._get_decoder_layer_modules()
        else:
            return self._get_encoder_layer_modules()
    
    def _get_all_layer_modules(self) -> List[nn.Module]:
        """Get all layer modules (encoder + decoder)."""
        encoder_layers = self._get_encoder_layer_modules()
        decoder_layers = self._get_decoder_layer_modules()
        return encoder_layers + decoder_layers

    def _get_encoder_layer_modules(self) -> List[nn.Module]:
        """Get encoder layer modules."""
        encoder = self._get_encoder()
        if hasattr(encoder, "block"):
            return list(encoder.block)
        elif hasattr(encoder, "layers"):
            return list(encoder.layers)
        else:
            raise ValueError("Cannot find encoder layers")

    def _get_decoder_layer_modules(self) -> List[nn.Module]:
        """Get decoder layer modules."""
        decoder = self._get_decoder()
        if hasattr(decoder, "block"):
            return list(decoder.block)
        elif hasattr(decoder, "layers"):
            return list(decoder.layers)
        else:
            raise ValueError("Cannot find decoder layers")

    def _get_attention_modules(self) -> List[nn.Module]:
        """Get all attention modules."""
        # Simplified - return self-attention from all layers
        layers = self._get_layer_modules()
        attns = []
        for layer in layers:
            if hasattr(layer, "layer"):  # T5
                attns.append(layer.layer[0])
            elif hasattr(layer, "self_attn"):  # BART
                attns.append(layer.self_attn)
            else:
                attns.append(layer)
        return attns

    def _get_mlp_modules(self) -> List[nn.Module]:
        """Get all MLP modules."""
        layers = self._get_layer_modules()
        mlps = []
        for layer in layers:
            if hasattr(layer, "layer"):  # T5
                mlps.append(layer.layer[-1])
            elif hasattr(layer, "fc1"):  # BART
                mlps.append(layer)
            else:
                mlps.append(layer)
        return mlps

    def _get_cross_attention_modules(self) -> List[nn.Module]:
        """Get cross-attention modules from decoder."""
        decoder_layers = self._get_decoder_layer_modules()
        cross_attns = []
        for layer in decoder_layers:
            if hasattr(layer, "layer") and len(layer.layer) > 1:  # T5
                cross_attns.append(layer.layer[1])
            elif hasattr(layer, "encoder_attn"):  # BART
                cross_attns.append(layer.encoder_attn)
        return cross_attns

    def forward(
        self,
        text: Union[str, List[str]],
        decoder_text: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Tuple[Tensor, EncDecActivationCache]:
        """
        Run forward pass for encoder-decoder model.
        
        Args:
            text: Encoder input text
            decoder_text: Decoder input text (optional, uses default if not provided)
        """
        self._enc_dec_cache = EncDecActivationCache()
        
        # Tokenize encoder input
        inputs = self._tokenize(text)
        
        # Tokenize decoder input if provided, or use a minimal decoder input
        if decoder_text is not None:
            decoder_inputs = self._tokenize(decoder_text)
            inputs["decoder_input_ids"] = decoder_inputs["input_ids"]
            inputs["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            # For T5/BART, we need some decoder input even for encoding-only tasks
            # Use the start token or a minimal input
            batch_size = inputs["input_ids"].shape[0]
            if hasattr(self.model.config, "decoder_start_token_id") and self.model.config.decoder_start_token_id is not None:
                start_token_id = self.model.config.decoder_start_token_id
            else:
                start_token_id = self.tokenizer.pad_token_id or 0
            inputs["decoder_input_ids"] = torch.full(
                (batch_size, 1), start_token_id, 
                dtype=torch.long, device=inputs["input_ids"].device
            )
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Populate the cache with hidden states for Tier 3 compatibility
        # Layer numbering depends on default_position:
        # - ENCODER_FINAL: layer_0 to layer_{num_encoder_layers-1} from encoder
        # - DECODER_FINAL: layer_0 to layer_{num_decoder_layers-1} from decoder
        
        if self._default_position == ExtractionPosition.DECODER_FINAL:
            # Use decoder hidden states
            if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states is not None:
                for i, hidden in enumerate(outputs.decoder_hidden_states[1:]):  # Skip embedding layer
                    if i < self.num_decoder_layers:
                        self._enc_dec_cache.hidden_states[f"layer_{i}"] = hidden
                        self._enc_dec_cache.decoder_hidden[f"layer_{i}"] = hidden
        else:
            # Use encoder hidden states (default for classification tasks)
            if hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states is not None:
                for i, hidden in enumerate(outputs.encoder_hidden_states[1:]):  # Skip embedding layer
                    if i < self.num_encoder_layers:
                        self._enc_dec_cache.hidden_states[f"layer_{i}"] = hidden
                        self._enc_dec_cache.encoder_hidden[f"layer_{i}"] = hidden
        
        # Get appropriate output
        if hasattr(outputs, "logits"):
            output = outputs.logits
        elif hasattr(outputs, "last_hidden_state"):
            output = outputs.last_hidden_state
        else:
            output = outputs.decoder_hidden_states[-1]
        
        return output, self._enc_dec_cache

    def get_embedding(
        self,
        text: Union[str, List[str]],
        layer: int = -1,
        position: Optional[ExtractionPosition] = None,
        decoder_text: Optional[Union[str, List[str]]] = None,
    ) -> Tensor:
        """
        Get embeddings from encoder, decoder, or cross-attention.
        
        Args:
            text: Encoder input text
            layer: Layer index (negative counts from end)
            position: Where to extract (encoder_final, decoder_final, cross_attn)
            decoder_text: Decoder input (required for decoder positions)
        """
        if position is None:
            position = self.default_position
        
        inputs = self._tokenize(text)
        
        # T5/BART always need decoder_input_ids, even for encoder-only extraction
        batch_size = inputs["input_ids"].shape[0]
        if hasattr(self.model.config, "decoder_start_token_id") and self.model.config.decoder_start_token_id is not None:
            start_token_id = self.model.config.decoder_start_token_id
        elif hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            start_token_id = self.tokenizer.pad_token_id
        else:
            start_token_id = 0
        
        # Always provide decoder input
        if decoder_text is not None:
            decoder_inputs = self._tokenize(decoder_text)
            inputs["decoder_input_ids"] = decoder_inputs["input_ids"]
            inputs["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            # Minimal decoder input (just start token)
            inputs["decoder_input_ids"] = torch.full(
                (batch_size, 1), start_token_id,
                dtype=torch.long, device=inputs["input_ids"].device
            )
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Handle layer indexing - need to account for embedding layer
        if position == ExtractionPosition.ENCODER_FINAL:
            # encoder_hidden_states includes embedding layer at index 0
            num_states = len(outputs.encoder_hidden_states)
            if layer < 0:
                layer = num_states + layer
            layer = min(layer, num_states - 1)
            hidden_states = outputs.encoder_hidden_states[layer]
            return self._extract_at_position(
                hidden_states,
                ExtractionPosition.MEAN_ALL,  # Encoder uses mean
                attention_mask=inputs.get("attention_mask"),
            )
        
        elif position == ExtractionPosition.DECODER_FINAL:
            num_states = len(outputs.decoder_hidden_states)
            if layer < 0:
                layer = num_states + layer
            layer = min(layer, num_states - 1)
            hidden_states = outputs.decoder_hidden_states[layer]
            return self._extract_at_position(
                hidden_states,
                ExtractionPosition.FINAL_TOKEN,  # Decoder uses final
                attention_mask=inputs.get("decoder_attention_mask"),
            )
        
        else:
            raise ValueError(f"Unsupported position for enc-dec: {position}")


@register_model("t5-small")
class T5SmallWrapper(EncoderDecoderWrapper):
    """T5 Small wrapper."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        position: ExtractionPosition = ExtractionPosition.ENCODER_FINAL,
        **kwargs
    ):
        super().__init__(
            model_name="t5-small",
            device=device,
            model_class=T5ForConditionalGeneration,
            default_position=position,
            **kwargs
        )


@register_model("t5-base")
class T5BaseWrapper(EncoderDecoderWrapper):
    """T5 Base wrapper."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        position: ExtractionPosition = ExtractionPosition.ENCODER_FINAL,
        **kwargs
    ):
        super().__init__(
            model_name="t5-base",
            device=device,
            model_class=T5ForConditionalGeneration,
            default_position=position,
            **kwargs
        )


@register_model("t5-large")
class T5LargeWrapper(EncoderDecoderWrapper):
    """T5 Large wrapper."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        position: ExtractionPosition = ExtractionPosition.ENCODER_FINAL,
        **kwargs
    ):
        super().__init__(
            model_name="t5-large",
            device=device,
            model_class=T5ForConditionalGeneration,
            default_position=position,
            **kwargs
        )


@register_model("facebook/bart-base")
class BartBaseWrapper(EncoderDecoderWrapper):
    """BART Base wrapper."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        position: ExtractionPosition = ExtractionPosition.ENCODER_FINAL,
        **kwargs
    ):
        super().__init__(
            model_name="facebook/bart-base",
            device=device,
            model_class=BartForConditionalGeneration,
            default_position=position,
            **kwargs
        )


@register_model("facebook/bart-large")
class BartLargeWrapper(EncoderDecoderWrapper):
    """BART Large wrapper."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        position: ExtractionPosition = ExtractionPosition.ENCODER_FINAL,
        **kwargs
    ):
        super().__init__(
            model_name="facebook/bart-large",
            device=device,
            model_class=BartForConditionalGeneration,
            default_position=position,
            **kwargs
        )


# Factory function
def create_enc_dec_wrapper(
    model_name: str,
    device: Optional[str] = None,
    position: ExtractionPosition = ExtractionPosition.ENCODER_FINAL,
    **kwargs
) -> EncoderDecoderWrapper:
    """
    Create an encoder-decoder wrapper for any HuggingFace seq2seq model.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        position: Default extraction position
        
    Returns:
        EncoderDecoderWrapper instance
    """
    wrapper = EncoderDecoderWrapper(
        model_name=model_name,
        device=device,
        model_class=AutoModelForSeq2SeqLM,
        default_position=position,
        **kwargs
    )
    return wrapper
