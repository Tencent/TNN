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

from utils import cmd
from utils import checker
from utils import return_code
from onnx_converter import onnx2tnn
from utils import align_model

from converter import logging

import os
import sys


def hack_name(names: str):
    hacked_names = ""
    name_list = names.split(';')
    for name in name_list:
        if name.endswith(":0"):
            hacked_names = hacked_names + name + ","
        else:
            hacked_names = hacked_names + name + ":0,"
    return hacked_names[:-1]

def process_input_names(input_names : str):
    split = input_names.split(";")
    name_list = []
    shape_list = []
    for item in split:
        temp = item.split("[")
        if not temp[0].endswith(":0"):
            temp[0] += ":0"
        name_list.append(temp[0])
        if len(temp) > 1:
            shape_list.append("[" + temp[1])
        else:
            shape_list.append("")

    inputs = ""
    inputs_as_nchw = ""
    for name, shape in zip(name_list, shape_list):
        inputs += (name + shape + ",")
        inputs_as_nchw += (name + ",")

    return inputs[:-1], inputs_as_nchw[:-1]

def tf2onnx(tf_path, input_names, output_name, onnx_path, not_fold_const=False):
    work_dir = "./"
    inputs, inputs_as_nchw = process_input_names(input_names)
    command = "python3 -m tf2onnx.convert  --graphdef " + tf_path

    command = command + " --inputs " + inputs
    command = command + " --inputs-as-nchw " + inputs_as_nchw

    command = command + " --outputs " + hack_name(output_name)
    command = command + " --output " + onnx_path
    command = command + " --opset 11"
    if not_fold_const is False:
        command = command + " --fold_const"

    logging.debug(command)
    result = cmd.run(command, work_dir=work_dir)
    if result == 0:
        return True
    else:
        return False


def convert(tf_path, input_names, output_names, output_dir, version, optimize, half, align=False, not_fold_const=False,
            input_path=None, refer_path=None):
    logging.info("Converter Tensorflow to TNN model\n")
    checker.check_file_exist(tf_path)
    model_name = os.path.basename(tf_path)
    if output_dir is None or not os.path.isdir(output_dir):
        output_dir = os.path.dirname(tf_path)
    checker.check_file_exist(output_dir)
    model_name = model_name[:-len(".pb")]
    onnx_path = os.path.join(output_dir, model_name + ".onnx")
    if tf2onnx(tf_path, input_names, output_names, onnx_path, not_fold_const) is False:
        logging.error("Oh No, tf2onnx failed :(\n")
        sys.exit(return_code.CONVERT_FAILED)
    else:
        logging.info("Convert TensorFlow to ONNX model succeed!\n")
    if version is None:
        version = "v1.0"
    checker.check_file_exist(onnx_path)
    onnx2tnn.convert(onnx_path, output_dir, version, optimize, half)

    if align is True:
        proto_suffix = '.tnnproto'
        model_suffix = '.tnnmodel'
        onnx_base_name = os.path.basename(onnx_path)
        if optimize is True:
            tnn_proto_name = onnx_base_name[:-len('.onnx')] + '.opt' + proto_suffix
            tnn_model_name = onnx_base_name[:-len('.onnx')] + '.opt' + model_suffix
        else:
            tnn_proto_name = onnx_base_name[:-len('.onnx')] + proto_suffix
            tnn_model_name = onnx_base_name[:-len('.onnx')] + model_suffix
        tnn_proto_path = os.path.join(output_dir, tnn_proto_name)
        tnn_model_path = os.path.join(output_dir, tnn_model_name)
        align_model.align_model(onnx_path, tnn_proto_path, tnn_model_path, input_path, refer_path)
