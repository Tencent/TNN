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

import onnx
from typing import *
from onnx import helper
from typing import *
import ctypes
import src.c2oObject as Node
import math


def create_custom_node(type_name: Text,
                       inputs: Sequence[Text],
                       outputs: Sequence[Text],
                       attributes: Dict) -> onnx.NodeProto:
    node = helper.make_node(type_name, inputs, outputs, **attributes)
    print(format(node))
    return node


def create_priorbox_attributes(layer) -> Dict:
    min_sizes = layer.prior_box_param.min_size
    max_sizes = layer.prior_box_param.max_size

    # onnx attributes does not support bool type
    flip = 1 if layer.prior_box_param.flip else 0
    clip = 1 if layer.prior_box_param.clip else 0

    aspect_ratio_tmp = layer.prior_box_param.aspect_ratio
    # get aspect ratio
    aspect_ratios = [1.0]
    for item in aspect_ratio_tmp:
        already_exist = False
        for i in range(len(aspect_ratios)):
            if math.fabs(item - aspect_ratios[i]) < 1e-6:
                already_exist = True
        if already_exist is False:
            aspect_ratios.append(item)
        if flip == 1:
            aspect_ratios.append(1. / item)

    # get variances     variances_tmp: List[float]
    variances = []
    if len(layer.prior_box_param.variance) > 1:
        assert len(layer.prior_box_param.variance) == 4
        variances = layer.prior_box_param.variance
    elif len(layer.prior_box_param.variance) == 1:
        variances = layer.prior_box_param.variance
    else:
        # set default to 0.1
        variances.append(0.1)

    # get image size
    img_sizes = [0, 0]
    if layer.prior_box_param.img_size != 0:
        img_sizes = [layer.prior_box_param.img_size, layer.prior_box_param.img_size]
    elif (layer.prior_box_param.img_h != 0) and (layer.prior_box_param.img_w != 0):
        # be careful the order: [img_w, img_h]
        img_sizes = [layer.prior_box_param.img_w, layer.prior_box_param.img_h]

    # get step
    steps = [0.0, 0.0]
    if layer.prior_box_param.step != 0:
        steps = [layer.prior_box_param.step, layer.prior_box_param.step]
    elif (layer.prior_box_param.step_h != 0) and (layer.prior_box_param.step_w != 0):
        # be careful the order: [step_w, step_h]
        steps = [layer.prior_box_param.step_w, layer.prior_box_param.step_h]

    offset = layer.prior_box_param.offset

    attributes = {
        'min_sizes': min_sizes,
        'max_sizes': max_sizes,
        'clip': clip,
        'flip': flip,
        'variances': variances,
        'aspect_ratios': aspect_ratios,
        'img_sizes': img_sizes,
        'steps': steps,
        'offset': offset
    }
    return attributes


def caculate_output_shape(layer, input_shape: List, attributes: Dict) -> List:
    width = input_shape[0][2]
    height = input_shape[0][3]
    aspect_ratios = attributes.get('aspect_ratios')
    min_sizes = attributes.get('min_sizes')
    num_priors = len(aspect_ratios) * len(min_sizes)
    max_sizes = attributes.get('max_sizes')
    for max_size in max_sizes:
        if max_size > 0:
            num_priors = num_priors + 1

    return [[1, 2, width * height * num_priors * 4]]


def create_priorbox_node(layer,
                         node_name: str,
                         inputs_name: List[str],
                         outputs_name: List[str],
                         inputs_shape: List, ) -> onnx.NodeProto:
    attributes = create_priorbox_attributes(layer)

    outputs_shape = caculate_output_shape(layer, inputs_shape, attributes)
    node = Node.c2oNode(layer, node_name, "PriorBox",
                        inputs_name, outputs_name,
                        inputs_shape, outputs_shape,
                        attributes)
    return node
