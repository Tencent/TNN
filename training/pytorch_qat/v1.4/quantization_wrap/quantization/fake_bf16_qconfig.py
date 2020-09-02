from __future__ import absolute_import, division, print_function, unicode_literals
# from torch.quantization.qconfig import QConfig
from collections import namedtuple
from .fake_bf16 import default_fake_bf16_weight, default_fake_bf16_activation

class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    def __new__(cls, activation, weight):
        return super(QConfig, cls).__new__(cls, activation, weight)


def get_bf16_qconfig():
    return QConfig(activation=default_fake_bf16_activation,
                          weight=default_fake_bf16_weight)

default_fake_bf16_qconfig = QConfig(activation=default_fake_bf16_activation,
                          weight=default_fake_bf16_weight)
