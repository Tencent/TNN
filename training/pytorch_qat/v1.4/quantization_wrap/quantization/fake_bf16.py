from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from functools import partial, wraps


def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.
    """
    # @wraps(_with_args)
    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r

def fake_bf16(x, export=False):
    if not export:
        x = x.bfloat16()
    x = x.to(torch.float32)
    return x

class FakeQuantize(Module):
    r""" 
        Simulate the quantize and dequantize operations in training time.
    """
    def __init__(self):
        super(FakeQuantize, self).__init__()
        self.fake_quant_enabled = True
        self.export = False

    def enable_fake_quant(self, enabled=True):
        self.fake_quant_enabled = enabled
        return self

    def disable_fake_quant(self):
        return self.enable_fake_quant(False)

    def enable_export(self, enabled=True):
        self.export = enabled
        return self

    def disable_export(self):
        return self.enable_export(False)

    def forward(self, x):
        if self.fake_quant_enabled:
            x = fake_bf16(x, self.export)
        return x

    with_args = classmethod(_with_args)

default_fake_bf16_weight = FakeQuantize.with_args()

default_fake_bf16_activation = FakeQuantize.with_args()

def disable_fake_quant(mod):
    if type(mod) == FakeQuantize:
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    if type(mod) == FakeQuantize:
        mod.enable_fake_quant()

def enable_export(mod):
    if type(mod) == FakeQuantize:
        mod.enable_export()

def disable_export(mod):
    if type(mod) == FakeQuantize:
        mod.disable_export()
