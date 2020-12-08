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

import numpy as np


def is_tf2onnx(onnx_path: str) -> bool:
    model = onnx.load(onnx_path)
    if model.producer_name == "tf2onnx":
        return True

    return False


def get_output_info(onnx_output_path: str) -> dict:
    raw_data = np.loadtxt(onnx_output_path, dtype=np.str, delimiter="\n")
    num_data = int(raw_data[0])
    raw_data = raw_data[1:]
    output_info = {}
    for i in range(num_data):
        info = raw_data[0].split(" ")
        name = info[0]
        dim_size = int(info[1])
        data_size = 1
        for item in info[2: -1]:
            data_size *= int(item)
        output_info[name] = dim_size
        raw_data = raw_data[data_size + 1:]

    return output_info


def find_target_layer(content: np.ndarray, target_name: str):
    for idx, layer in enumerate(content):
        if target_name in layer:
            name = layer.split(" ")[1]
            return idx, name

    raise Exception("Conversion failed");


def generate_transpose(input_name: str, output_name: str, dim_size: int) -> str:
    if dim_size == 3:
        permute_param = "0 2 1 3"
    else:
        permute_param = "0 3 1 2"

    name = "Transpose{}" .format(input_name)
    return "\"Permute {} 1 1 {} {} 4 {} ,\"" .format(name, input_name, output_name, permute_param)


def replace_output_name(layer_info: str, src_name: str, dst_name: str):
    layer_info_ = layer_info.split(" ")
    input_cnt = int(layer_info_[2])
    output_cnt = int(layer_info_[3])
    for i in range(4 + input_cnt, 4 + input_cnt + output_cnt):
        if layer_info_[i] == src_name:
            layer_info_[i] = dst_name

    return " " .join(layer_info_)


def fix_tnn_output(onnx_path: str, tnnproto_path: str, output_path: str):
    if not is_tf2onnx(onnx_path):
        return

    output_info = get_output_info(output_path)

    tnnproto = np.loadtxt(tnnproto_path, dtype=np.str, delimiter="\n")
    offset = 5
    for output_name, dim_size in output_info.items():
        if dim_size == 3 or dim_size == 4:
            idx, name = find_target_layer(tnnproto[offset:], output_name)
            inner_output_name = "__" + output_name
            tnnproto[offset + idx] = replace_output_name(tnnproto[offset + idx], output_name, inner_output_name)
            tnnproto = np.append(tnnproto, generate_transpose(inner_output_name, output_name, dim_size))

    layer_cnt = tnnproto[offset:].shape[0]
    tnnproto[4] = "\" {} ,\"" .format(layer_cnt)
    np.savetxt(tnnproto_path, tnnproto, delimiter="\n", fmt="%s")