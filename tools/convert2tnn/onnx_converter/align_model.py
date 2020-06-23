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
from utils import checker
from utils import parse_path
from utils import cmd
from utils import data
from utils import convert_name

import linecache
import math
import os
import onnxruntime

import numpy as np


def run_tnn_model_check(proto_path, model_path, input_path, reference_output_path):
    cmd.run("pwd")
    relative_path = "bin/model_check"
    model_check_path = parse_path.parse_path(relative_path)
    checker.check_file_exist(model_check_path)
    command = model_check_path + " -p  " + proto_path + " -m " + \
        model_path + " -i " + input_path + " -f " + reference_output_path + " -d NAIVE"

    print(command)
    cmd.run(command)
    return


def run_onnx_input_dict(model_path, input_feed, output_path_list):
    session = onnxruntime.InferenceSession(model_path)

    if len(input_feed) != len(session.get_inputs()):
        print("The input of model = {}, the length of inputs = {}, they should be equal"
              .format(len(session.get_inputs()), len(input_feed)))
        return False

    for i, output in enumerate(session.get_outputs()):
        output_name = output.name
        pred = session.run([output_name, ], input_feed)
        print("The output_name:\n{}:\n{}".format(
            output_name, np.shape(pred[0])))
        np.savetxt(output_path_list[i], pred[0].reshape(-1), fmt="%0.18f")

    return True


def run_onnx(model_path: str, input_path: str, input_info: dict) -> str:
    session = onnxruntime.InferenceSession(model_path)

    output_path = input_path
    deli = "/"
    if output_path[-1] == "/":
        output_path = output_path[:-1]
    output_path = deli.join(output_path.split("/")[:-1])
    output_path += "/output-onnx.txt"

    input_name, input_shape = list(input_info.items())[0]

    if type(input_shape[0]) is not int:
        input_shape[0] = 1

    input_data = np.loadtxt(input_path)
    input_data = input_data.astype(np.float32)
    input_data = np.reshape(input_data, input_shape)
    output_info = session.get_outputs()
    pred = session.run([], {input_name: input_data})
    with open(output_path, "w") as f:
        f.write("{}\n" .format(len(output_info)))
        cnt = 0
        for item in output_info:
            output_name = item.name
            output_shape = pred[cnt].shape

            description = "{} {} " .format(output_name, len(output_shape))
            for dim in output_shape:
                description += "{} " .format(dim)
            f.write(description + "\n")
            np.savetxt(f, pred[cnt].reshape(-1), fmt="%0.18f")
            cnt += 1

    return output_path


def get_input_shape_from_onnx(onnx_path) -> dict:
    session = onnxruntime.InferenceSession(onnx_path)
    input_info: dict = {}
    for ip in session.get_inputs():
        name = ip.name
        shape = ip.shape
        if type(shape[0]) is not int:
            shape[0] = 1
        input_info.update({name: shape})
    return input_info


def get_input_shape_from_tnn(tnn_proto_path):
    input_info: dict = {}
    line = linecache.getline(tnn_proto_path, 2).strip(
        '\n').strip('\"').strip(',')
    input_list = line.split(':')
    for input in input_list:
        name, n, c, h, w = input.strip(' ').split(' ')
        input_info.update({name: [int(n), int(c), int(h), int(w)]})
    return input_info


def print_not_align_message(reason):
    print("==================== Unfortunately============================\n")
    print("The onnx model not aligned with tnn model\n")
    print("the reason " + reason)
    exit(-1)


def print_align_message():
    print("====================Congratulations!==========================\n")
    print("the onnx model aligned whit tnn model\n")


def check_input_info(onnx_input_info: dict, tnn_input_info: dict):
    if len(onnx_input_info) != len(tnn_input_info):
        print_not_align_message("onnx input size != tnn input size")
    for name, onnx_shape in onnx_input_info.items():
        tnn_name = convert_name.onnx_name2tnn_name(name)
        tnn_shape = tnn_input_info[tnn_name]
        if type(onnx_shape[0]) is not int:
            onnx_shape[0] = 1
        if tnn_shape != onnx_shape:
            print_not_align_message(
                "the {}'s shape not equal! the onnx shape:{}, tnn shape: {}".format(name, str(onnx_shape),
                                                                                    str(tnn_shape)))
    
    print("check onnx input shape and tnn input shape align!\n")

def parse_input_names(input_names: str) -> dict:
    input_info = {}
    for inp in input_names.split(";"):
        name, shape_ = inp.split(":")
        shape = []
        for dim in shape_.split(","):
            shape.append(int(dim))

        shape[0] = 1

        input_info[name] = shape

    return input_info


def align_model(onnx_path: str, tnn_proto_path: str, tnn_model_path: str, input_file_path: str=None,
                refer_path: str = None, input_names: str = None) -> bool:
    """
    对 onnx 模型和 tnn 模型进行对齐.
    当前支持模型: 单输入,单输出;单输入,多输出;
    :param onnx_path:
    :param tnn_proto_path:
    :param tnn_model_path:
    :return:
    """
    checker.check_file_exist(tnn_proto_path)
    checker.check_file_exist(tnn_model_path)

    if input_names is not None:
        input_info = parse_input_names(input_names)

    # check input
    if input_names is not None:
        tnn_input_info = input_info
        onnx_input_info = input_info
    else:
        tnn_input_info = get_input_shape_from_tnn(tnn_proto_path)
        onnx_input_info = get_input_shape_from_onnx(onnx_path)
    check_input_info(onnx_input_info, tnn_input_info)

    if input_file_path is None:
        # generate data
        input_path = data.gene_random_data(onnx_input_info)
    else:
        if os.path.exists(input_file_path):
            input_path = input_file_path
        else:
            print("invalid input_file_path")
            exit(-1)

    if refer_path is None:
        reference_output_path = run_onnx(onnx_path, input_path, onnx_input_info)
    else:
        if os.path.exists(refer_path):
            reference_output_path = refer_path
        else:
            print("invalid refer_path")
            exit(-1)

    run_tnn_model_check(tnn_proto_path, tnn_model_path, input_path, reference_output_path)
    
    if input_file_path is None and os.path.exists(input_path):
        data.clean_temp_data(os.path.dirname(input_path))
    if refer_path is None and os.path.exists(reference_output_path):
        data.clean_temp_data(reference_output_path)
    
    return True

