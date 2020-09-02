# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# danerli modify 2020.8.17


"""Logic to update a Pytorch model graph with quantization operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections


class InputToOps(object):
    """Holds a mapping from tensor's name to ops that take it as input."""
    def __init__(self, graph):
        """Initializes mapping from op's name to ops that take it.
        Helps find edges between ops faster and avoids iterating over the whole
        graph.   The mapping is of type Dict[str, Set[tf.Operation]].
        Note: while inserting operations into the graph, we do not update the
        mapping, assuming that insertion points in the graph are never adjacent.
        With that restriction, an out of date mapping still works fine.
        Args:
          graph: Graph to process.
        """
        self.mapping_consumer = collections.defaultdict(set)
        self.mapping_producer = collections.defaultdict(set)
        for op in graph.nodes():
            for op_input in list(op.inputs()):
                self.mapping_consumer[op_input].add(op)
            for op_output in list(op.outputs()):
                self.mapping_producer[op_output].add(op)

    def consumer_operations(self, producer_op):
        """Looks through outputs of producer_op, finds ops that take them as input.
        Args:
          producer_op: Operation containing outputs to process.
        Returns:
          A Set[Operation] containing all operations taking input from producer_op
            outputs.
        """
        result = set()
        for inp in list(producer_op.outputs()):
          result.update(self.mapping_consumer[inp])
        return result

    def producer_operations(self, consumer_op):
        """Looks through outputs of producer_op, finds ops that take them as input.
        Args:
          consumer_op: Operation containing outputs to process.
        Returns:
          A Set[Operation] containing all operations taking input from producer_op
            outputs.
        """
        result = set()
        for oup in list(consumer_op.inputs()):
            result.update(self.mapping_producer[oup])
        return result