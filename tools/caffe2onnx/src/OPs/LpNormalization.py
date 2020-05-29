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

import src.c2oObject as Node
from typing import Dict


def create_attribute(layer):
    attribute: Dict = {
        'axis': 1,
        'p': 2
    }
    return attribute


def get_node_output(input_shape):
    output_shape = input_shape
    return output_shape


def create_Lp_Normalization(layer, node_name, input_name, output_name, input_shape):
    attribute = create_attribute(layer)
    output_shape = get_node_output(input_shape)

    node = Node.c2oNode(layer, node_name, "LpNormalization", input_name, output_name, input_shape, output_shape,
                        attribute)
    return node
