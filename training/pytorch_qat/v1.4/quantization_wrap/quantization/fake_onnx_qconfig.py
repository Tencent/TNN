from __future__ import absolute_import, division, print_function, unicode_literals
# from torch.quantization.qconfig import QConfig
from collections import namedtuple
from .fake_onnx import default_fake_onnx_weight, default_fake_onnx_activation

class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    def __new__(cls, activation, weight):
        return super(QConfig, cls).__new__(cls, activation, weight)


def get_onnx_qconfig():
    return QConfig(activation=default_fake_onnx_activation,
                          weight=default_fake_onnx_weight)

default_fake_onnx_qconfig = QConfig(activation=default_fake_onnx_activation,
                          weight=default_fake_onnx_weight)
