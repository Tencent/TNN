# BasePattern for subgraph search
# danerli add 2020.8.3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class BasePattern(object):

  def build_pattern(self):
      raise NotImplementedError()

  def match_pattern(self, graph):
      raise NotImplementedError()

  def warp_result(self, match_result):
      """Populates a layer match object containing ops/tensors for folding BNs.

      Args:
        match_result: Matched result from graph matcher

      Returns:
        layer_op: Matching conv/fc op prior to batch norm
        BatchNormMatch: _BatchNormMatch containing all required batch norm
        parameters.
      """
      raise NotImplementedError()
