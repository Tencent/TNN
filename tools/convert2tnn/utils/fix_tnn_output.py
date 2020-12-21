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

def get_output_info(tnnproto: list) -> list:
    output_info = tnnproto[3]
    output_name = output_info.split("\"")[1]
    output_name = output_name.split(" ")[:-1]
    return output_name


def find_target_layer(content: list, target_name: str):
    for idx, layer in enumerate(content):
        if target_name in layer:
            name = layer.split(" ")[1]
            return idx, name

    raise Exception("Conversion failed")


def generate_transpose(input_name: str, output_name: str) -> str:
    return "\"Permute {} 1 1 {} {} 4 0 3 1 2 ,\"\n" .format(output_name, input_name, output_name)


def replace_output_name(layer_info: str, src_name: str, dst_name: str):
    layer_info_ = layer_info.split(" ")
    input_cnt = int(layer_info_[2])
    output_cnt = int(layer_info_[3])
    for i in range(4 + input_cnt, 4 + input_cnt + output_cnt):
        if layer_info_[i] == src_name:
            layer_info_[i] = dst_name

    return " " .join(layer_info_)


def fix_tnn_output(tnnproto_path: str):
    with open(tnnproto_path) as f:
        tnnproto = f.readlines()
    output_info = get_output_info(tnnproto)
    offset = 5
    add_layers = []
    for output_name in output_info:
        idx, name = find_target_layer(tnnproto[offset:], output_name)
        inner_output_name = output_name + "_fix_output_name_from_tf2onnx"
        add_layers.append(inner_output_name)
        tnnproto[offset + idx] = replace_output_name(tnnproto[offset + idx], output_name, inner_output_name)
        tnnproto.append(generate_transpose(inner_output_name, output_name))

    # fix op
    info = tnnproto[0].split(" ")
    total_ops = int(info[1]) + len(add_layers)
    info[1] = str(total_ops)
    tnnproto[0] = " ".join(info)

    op_name_list = tnnproto[2].split(" ")
    new_ops_name_list = op_name_list[:1] + add_layers + op_name_list[1:]
    new_ops_name = " ".join(new_ops_name_list)
    tnnproto[2] = new_ops_name

    # fix layer cnt
    layer_cnt = len(tnnproto[offset:])
    tnnproto[4] = "\" {} ,\"\n" .format(layer_cnt)
    with open(tnnproto_path, "w") as f:
        f.writelines(tnnproto)
