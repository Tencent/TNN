from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from quantization_wrap.utils import conv_bn_pattern, conv_bn_relu_pattern, aten_op_pattern


def conv_bn_match(graph):
  conv_bn_match = conv_bn_pattern.ConvBnPattern()
  layer_matches = conv_bn_match.match_pattern(graph)
  return layer_matches

def conv_bn_relu_match(graph):
  conv_bn_relu_match = conv_bn_relu_pattern.ConvBnReluPattern()
  layer_matches = conv_bn_relu_match.match_pattern(graph)
  return layer_matches

def aten_op_match(graph):
  aten_op_match = aten_op_pattern.AtenOpPattern()
  aten_op_matches = aten_op_match.match_pattern(graph)
  return aten_op_matches
