"""
Architecture wrappers for different transformer types.
"""

from mechinterp.architectures.base import HookedModelWrapper
from mechinterp.architectures.encoder import (
    EncoderWrapper,
    BertBaseWrapper,
    BertLargeWrapper,
    RobertaBaseWrapper,
    RobertaLargeWrapper,
    create_encoder_wrapper,
)
from mechinterp.architectures.decoder import (
    DecoderWrapper,
    GPT2SmallWrapper,
    GPT2MediumWrapper,
    GPT2LargeWrapper,
    create_decoder_wrapper,
)
from mechinterp.architectures.enc_dec import (
    EncoderDecoderWrapper,
    EncDecActivationCache,
    T5SmallWrapper,
    T5BaseWrapper,
    T5LargeWrapper,
    BartBaseWrapper,
    BartLargeWrapper,
    create_enc_dec_wrapper,
)

__all__ = [
    # Base
    "HookedModelWrapper",
    # Encoder
    "EncoderWrapper",
    "BertBaseWrapper",
    "BertLargeWrapper",
    "RobertaBaseWrapper",
    "RobertaLargeWrapper",
    "create_encoder_wrapper",
    # Decoder
    "DecoderWrapper",
    "GPT2SmallWrapper",
    "GPT2MediumWrapper",
    "GPT2LargeWrapper",
    "create_decoder_wrapper",
    # Encoder-Decoder
    "EncoderDecoderWrapper",
    "EncDecActivationCache",
    "T5SmallWrapper",
    "T5BaseWrapper",
    "T5LargeWrapper",
    "BartBaseWrapper",
    "BartLargeWrapper",
    "create_enc_dec_wrapper",
]
