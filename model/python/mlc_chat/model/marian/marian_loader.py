"""
This file specifies how MLC's Whisper parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_chat.loader import ExternMapping
from mlc_chat.quantization import Quantization

from .marian_model import MarianConfig, MarianMT
from .marian_quantization import awq_quant


def huggingface(model_config: MarianConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : WhisperConfig
        The configuration of the GPT-2 model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = MarianMT(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    _, _named_params = model.export_tvm(spec=model.get_default_spec())
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    
    mlc_name = ["lm_head.weight"]
    src_name = ["model.shared.weight"]
    for mlc_name, src_name in zip(mlc_name, src_name):
        mapping.add_mapping(
            mlc_name,
            [src_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )
       

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    return mapping


def awq(model_config: MarianConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of AWQ parameters.
    Parameters
    ----------
    model_config : WhisperConfig
        The configuration of the Whisper model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to AWQ.
    """
    model, _ = awq_quant(model_config, quantization)
    _, _named_params = model.export_tvm(spec=model.get_default_spec())
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    
    mlc_name = ["lm_head.weight"]
    src_name = ["model.shared.weight"]
    for mlc_name, src_name in zip(mlc_name, src_name):
        mapping.add_mapping(
            mlc_name,
            [src_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )
      

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
            )
    return mapping
