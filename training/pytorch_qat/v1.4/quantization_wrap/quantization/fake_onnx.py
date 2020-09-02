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


class FakeQuantize(Module):
    r""" 
        Simulate the quantize and dequantize operations in training time.
    """
    def __init__(self):
        super(FakeQuantize, self).__init__()
        self.fake_quant_enabled = True
        self.export = False
        self.dtype= torch.quint8
        self.scale = torch.tensor(1.0)
        self.zero_point = torch.tensor(0.0)

    def calculate_qparams(self):
        return self.scale, self.zero_point

    def set_qparams(self, scale, zero):
        self.scale = scale
        self.zero_point = zero

    def forward(self, x):
        return x

    with_args = classmethod(_with_args)

default_fake_onnx_weight = FakeQuantize.with_args()

default_fake_onnx_activation = FakeQuantize.with_args()

