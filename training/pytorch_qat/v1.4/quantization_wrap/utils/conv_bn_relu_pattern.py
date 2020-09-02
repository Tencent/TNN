# ConvBnReluPattern for CONV + BN + RELU subgraph search
# danerli add 2020.8.3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from quantization_wrap.utils import graph_matcher
from quantization_wrap.utils.conv_bn_pattern import ConvBnPattern

class ConvBnReluPattern(ConvBnPattern):

    def __init__(self):

        super().__init__()

        self.build_pattern()

    def build_pattern(self):

        super().build_pattern()

        self.relu_pattern = graph_matcher.OpTypePattern(
            'aten::relu_|aten::relu',
            inputs=[
                self.batch_norm_pattern
            ])

        self.relu_matcher = graph_matcher.GraphMatcher(self.relu_pattern)

    def _replace(self, conv, bn, relu):
        index = self.result.index([conv, bn])
        self.result[index] = [conv, bn, relu]

    def match_pattern(self, graph):

        # We use matched_layer_set to ensure that layers aren't matched multiple
        # times.

        for match_result in self.batch_norm_matcher.match_graph(graph):
            conv, bn = super().warp_result(match_result)
            self.result.append([conv, bn])

        for match_result in self.relu_matcher.match_graph(graph):

            conv, bn, relu = self.warp_result(match_result)

            if [conv, bn] in self.result:
                self._replace(conv, bn, relu)
            else:
                self.result.append([conv, bn, relu])

        return self.result

    def warp_result(self, match_result):
        """Populates a layer match object containing ops/tensors for folding BNs.

        Args:
        match_result: Matched result from graph matcher

        Returns:
            layer_op: Matching conv/fc op prior to batch norm
            ReluNormMatch: ReluhNormMatch containing all required relu
            parameters.
        """

        conv, bn = super().warp_result(match_result)

        relu_pattern = match_result.get_op(self.relu_pattern)
        relu = str(relu_pattern.scopeName()).split('/')[-1].replace('__module.', '')

        return conv, bn, relu

