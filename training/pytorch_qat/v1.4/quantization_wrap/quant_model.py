import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.intrinsic as nni
import torchvision
from torchvision.models.quantization.utils import _replace_relu
from torch.quantization.quantize import propagate_qconfig_, DEFAULT_QAT_MODULE_MAPPING
from torch.quantization.default_mappings import (DEFAULT_DYNAMIC_MODULE_MAPPING,
                               DEFAULT_MODULE_MAPPING,
                               DEFAULT_QAT_MODULE_MAPPING,
                               DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST)

def replace_relu(module):
    _replace_relu(module)


def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output
    """
    return self.activation_post_process(output)

def add_observer_(module):
    r"""Add observer for the leaf child of the module.

    This function insert observer module to all leaf child module that
    has a valid qconfig attribute.

    Args:
        module: input module with qconfig attributes for all the leaf modules that we want to quantize

    Return:
        None, module is modified inplace with added observer modules and forward_hooks
    """
    for child in module.children():
        if type(child) == nnq.FloatFunctional:
            if hasattr(child, 'qconfig') and child.qconfig is not None:
                child.activation_post_process = child.qconfig.activation()
        else:
            add_observer_(child)

    # Insert observers only for leaf nodes, note that this observer is for
    # the output of the module, for input QuantStub will observe them
    if hasattr(module, 'qconfig') and module.qconfig is not None and \
       len(module._modules) == 0 and not isinstance(module, torch.nn.Sequential):
        # observer and hook will be gone after we swap the module
        module.add_module('activation_post_process', module.qconfig.activation())
        module.register_forward_hook(_observer_forward_hook)

def prepare(model, qconfig_dict=None, inplace=False):
    r"""Prepares a copy of the model for quantization calibration or quantization-aware training.

    Quantization configuration can be passed as an `qconfig_dict` or assigned preemptively
    to individual submodules in `.qconfig` attribute.

    The model will be attached with observer or fake quant modules, and qconfig
    will be propagated.

    Args:
        model: input model to be modified in-place
        qconfig_dict: dictionary that maps from name or type of submodule to quantization
            configuration, qconfig applies to all submodules of a given
            module unless qconfig for the submodules are specified (when the
            submodule already has qconfig attribute)
        inplace: carry out model transformations in-place, the original module is mutated
    """
    if not inplace:
        model = copy.deepcopy(model)
    propagate_qconfig_(model)
    # sanity check common API misusage
    if not any(hasattr(m, 'qconfig') and m.qconfig for m in model.modules()):
        warnings.warn("None of the submodule got qconfig applied. Make sure you "
                      "passed correct configuration through `qconfig_dict` or "
                      "by assigning the `.qconfig` attribute directly on submodules")
    add_observer_(model)
    return model

def convert(module, mapping=None, inplace=False):
    r"""Converts the float module with observers (where we can get quantization
    parameters) to a quantized module.

    Args:
        module: calibrated module with observers
        mapping: a dictionary that maps from float module type to quantized
                 module type, can be overwrritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated

    """
    if mapping is None:
        mapping = DEFAULT_MODULE_MAPPING
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    # TODO(jerryzh): remove after deciding on the impl of intrinsic modules
    # This is required because intrinsic modules right now are implemented as
    # nn.Sequential and we don't want to swap their constituents
    SWAPPABLE_MODULES = (nni.ConvBn2d,
                         nni.ConvBnReLU2d,
                         nni.LinearReLU,
                         nni.ConvReLU2d)

    for name, mod in module.named_children():
        if type(mod) not in SWAPPABLE_MODULES:
            convert(mod, mapping, inplace=True)
        reassign[name] = swap_module(mod, mapping)

    for key, value in reassign.items():
        module._modules[key] = value

    return module

def swap_module(mod, mapping):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    # Always replace dequantstub with dequantize
    if hasattr(mod, 'qconfig') and mod.qconfig is not None or type(mod) == DeQuantStub:
        if type(mod) in mapping:
            new_mod = mapping[type(mod)].from_float(mod)
    return new_mod

def prepare_qat(model, mapping=None, inplace=False):
    r"""
    Prepares a copy of the model for quantization calibration or
    quantization-aware training and convers it to quantized version.

    Quantization configuration can be passed as an `qconfig_dict` or assigned
    preemptively to individual submodules in `.qconfig` attribute.

    Args:
        model: input model to be modified in-place
        mapping: dictionary that maps float modules to quantized modules to be
                 replaced.
        inplace: carry out model transformations in-place, the original module
                 is mutated
    """
    if mapping is None:
        mapping = DEFAULT_QAT_MODULE_MAPPING
    model = prepare(model, inplace=inplace)
    convert(model, mapping, inplace=True)
    return model