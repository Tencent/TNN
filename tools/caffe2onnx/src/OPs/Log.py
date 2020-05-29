# Tencent is pleased to support the open source community by making TNN available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Message that stores parameters used by LogLayer
# message LogParameter {
#   // LogLayer computes outputs y = log_base(shift + scale * x), for base > 0.
#   // Or if base is set to the default (-1), base is set to e,
#   // so y = ln(shift + scale * x) = log_e(shift + scale * x)
#   optional float base = 1 [default = -1.0];
#   optional float scale = 2 [default = 1.0];
#   optional float shift = 3 [default = 0.0];
# }

import src.c2oObject as Node


def get_log_output_shape(input_shape):
    output_shape = input_shape
    return output_shape


def create_log_node(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_log_output_shape(layer)

    node = Node.c2oNode(layer, node_name, 'Log', input_name, output_name, input_shape, output_shape)

    return node
