# ConvBnPattern for CONV + BN subgraph search
# danerli add 2020.8.3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from quantization_wrap.utils import graph_matcher
from quantization_wrap.utils.base_pattern import BasePattern

class ConvBnPattern(BasePattern):

    def __init__(self):

        super().__init__()

        self.build_pattern()

        self.conv = set()

        self.result = list()

    def build_pattern(self):

        self.input_pattern = graph_matcher.OpTypePattern('*')
        self.weight_pattern = graph_matcher.OpTypePattern('*')
        self.bias_pattern = graph_matcher.OpTypePattern('*')
        self.stride_pattern = graph_matcher.OpTypePattern('*', inputs=[graph_matcher.OpTypePattern('*'),
                                                                       graph_matcher.OpTypePattern('*')])
        self.padding_pattern = graph_matcher.OpTypePattern('*', inputs=[graph_matcher.OpTypePattern('*'),
                                                                        graph_matcher.OpTypePattern('*')])
        self.dilation_pattern = graph_matcher.OpTypePattern('*', inputs=[graph_matcher.OpTypePattern('*'),
                                                                         graph_matcher.OpTypePattern('*')])
        self.transposed_pattern = graph_matcher.OpTypePattern('*')
        self.output_padding_pattern = graph_matcher.OpTypePattern('*', inputs=[graph_matcher.OpTypePattern('*'),
                                                                               graph_matcher.OpTypePattern('*')])
        self.groups_pattern = graph_matcher.OpTypePattern('*')
        self.benchmarkCuDNN_pattern = graph_matcher.OpTypePattern('*')
        self.deterministicCuDNN_or_deterministic_pattern = graph_matcher.OpTypePattern('*')
        self.userEnabledCuDNN_pattern = graph_matcher.OpTypePattern('*')

        self._convolution_pattern = graph_matcher.OpTypePattern(
            'aten::_convolution',
            inputs=[
                self.input_pattern,
                self.weight_pattern,
                self.bias_pattern,
                self.stride_pattern,
                self.padding_pattern,
                self.dilation_pattern,
                self.transposed_pattern,
                self.output_padding_pattern,
                self.groups_pattern,
                self.benchmarkCuDNN_pattern,
                self.deterministicCuDNN_or_deterministic_pattern,
                self.userEnabledCuDNN_pattern
            ])

        self.input_reshaped = graph_matcher.OneofPattern([self._convolution_pattern])

        self.weight_ = graph_matcher.OpTypePattern('*')
        self.bias_ = graph_matcher.OpTypePattern('*')
        self.running_mean_ = graph_matcher.OpTypePattern('*')
        self.running_var_ = graph_matcher.OpTypePattern('*')
        self.use_input_stats = graph_matcher.OpTypePattern('*')
        self.momentum = graph_matcher.OpTypePattern('*')
        self.eps = graph_matcher.OpTypePattern('*')
        self.cudnn_enabled = graph_matcher.OpTypePattern('*')

        self.batch_norm_pattern = graph_matcher.OpTypePattern(
            'aten::batch_norm',
            inputs=[
                self.input_reshaped,
                self.weight_,
                self.bias_,
                self.running_mean_,
                self.running_var_,
                self.use_input_stats,
                self.momentum,
                self.eps,
                self.cudnn_enabled
            ])

        self.batch_norm_matcher = graph_matcher.GraphMatcher(self.batch_norm_pattern)

        self.relu_pattern = graph_matcher.OpTypePattern(
            'aten::relu_|aten::relu',
            inputs=[
                self.batch_norm_pattern
            ])

        self.relu_matcher = graph_matcher.GraphMatcher(self.relu_pattern)

    def match_pattern(self, graph):

        # We use matched_layer_set to ensure that layers aren't matched multiple
        # times.

        for match_result in self.batch_norm_matcher.match_graph(graph):
            conv, bn = self.warp_result(match_result)
            self.result.append([conv, bn])

        return self.result

    def warp_result(self, match_result):
        """Populates a layer match object containing ops/tensors for folding BNs.

        Args:
        match_result: Matched result from graph matcher

        Returns:
            layer_op: Matching conv/fc op prior to batch norm
            BatchNormMatch: _BatchNormMatch containing all required batch norm
            parameters.
        """
        _convolution_pattern = match_result.get_op(self._convolution_pattern)
        conv = str(_convolution_pattern.scopeName()).split('/')[-1].replace('__module.', '')

        batch_norm_pattern = match_result.get_op(self.batch_norm_pattern)
        bn = str(batch_norm_pattern.scopeName()).split('/')[-1].replace('__module.', '')

        return conv, bn
