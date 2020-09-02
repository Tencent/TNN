# AtenOpPattern for AtenOp subgraph search
# danerli add 2020.8.3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from quantization_wrap.utils import graph_matcher
from quantization_wrap.utils.base_pattern import BasePattern

class AtenOpPattern(BasePattern):

    def __init__(self):

        super().__init__()

        self.conv_bn_relu = False

        self.build_pattern()

        self.conv = set()

        self.conv_result = list()
        self.bn_result = list()
        self.relu_result = list()

    def build_pattern(self):

        self._convolution_pattern = graph_matcher.OpTypePattern('aten::_convolution')

        self.batch_norm_pattern = graph_matcher.OpTypePattern('aten::batch_norm')

        self.relu_pattern = graph_matcher.OpTypePattern('aten::relu_|aten::relu')

        self._convolution_matcher = graph_matcher.GraphMatcher(self._convolution_pattern)

        self.batch_norm_matcher = graph_matcher.GraphMatcher(self.batch_norm_pattern)

        self.relu_matcher = graph_matcher.GraphMatcher(self.relu_pattern)

    def match_pattern(self, graph):

        # We use matched_layer_set to ensure that layers aren't matched multiple
        # times.

        for match_result in self._convolution_matcher.match_graph(graph):
            _convolution_pattern = match_result.get_op(self._convolution_pattern)
            conv = str(_convolution_pattern.scopeName()).split('/')[-1].replace('__module.', '')
            self.conv_result.append([conv])

        for match_result in self.batch_norm_matcher.match_graph(graph):
            batch_norm_pattern = match_result.get_op(self.batch_norm_pattern)
            bn = str(batch_norm_pattern.scopeName()).split('/')[-1].replace('__module.', '')
            self.bn_result.append([bn])

        for match_result in self.relu_matcher.match_graph(graph):
            relu_pattern = match_result.get_op(self.relu_pattern)
            relu = str(relu_pattern.scopeName()).split('/')[-1].replace('__module.', '')
            self.relu_result.append([relu])

        return {'conv': self.conv_result, 'bn': self.bn_result, 'relu': self.relu_result}

    def enable_conv_bn_relu(self, enabled=True):
        self.conv_bn_relu = enabled
        return self

    def disable_conv_bn_relu(self):
        return self.enable_conv_bn_relu(False)
