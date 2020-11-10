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
from utils import return_code

from converter import logging

import linecache
import math
import os
import onnxruntime
import sys

import tensorflow as tf
import numpy as np


def run_tnn_model_check(proto_path, model_path, input_path, reference_output_path, is_tflite=False):
    cmd.run("pwd")
    relative_path = "bin/model_check"
    model_check_path = parse_path.parse_path(relative_path)
    checker.check_file_exist(model_check_path)
    command = model_check_path + " -p  " + proto_path + " -m " + \
        model_path + " -i " + input_path + " -f " + reference_output_path + " -d NAIVE"

    logging.debug(command)
    ret = cmd.run(command)

    if ret == 0:
        print_align_message(is_tflite)
    else:
        print_not_align_message(None, is_tflie)

    return


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

def run_tflite(model_path: str, input_path: str, input_info: dict) -> str:
    output_path = input_path
    deli = "/"
    if output_path[-1] == "":
        output_path = output_path[:-1]
    output_path = deli.join(output_path.split("/")[:-1])
    output_path += "/output-onnx.txt"
    input_name, input_shape = list(input_info.items())[0]
    if type(input_shape[0]) is not int:
        input_shape[0] = 1
    nchw = [1,1,1,1]
    nchw[0] = input_shape[0]
    nchw[1] = input_shape[3]
    nchw[2] = input_shape[1]
    nchw[3] = input_shape[2]
    input_data = np.loadtxt(input_path)
    input_data = input_data.astype(np.float32)
    input_data = np.reshape(input_data, nchw) # input_shape is nchw
    input_data = np.transpose(input_data, (0, 2, 3, 1)) # transpose to nhwc
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    with open(output_path, "w") as f:
        f.write("{}\n" .format(len(output_details)))
        for item in output_details:
           output_name = item["name"]
           index = item["index"]
           output_data = interpreter.get_tensor(index)

           shape = list(output_data.shape)
           while len(shape) < 4:
               shape.insert(-1, 1)
               output_data = output_data.reshape(*shape)

           output_data = np.transpose(output_data, (0, 3, 1, 2)) # transpose result from nhwc to nchw
           output_shape = output_data.shape
           description = "{} {} " .format(output_name, len(output_shape))
           for dim in output_shape:
             description += "{} " .format(dim)
           f.write(description + "\n")
           np.savetxt(f, output_data.reshape(-1), fmt="%0.18f")
    return output_path


def get_input_shape_from_onnx(onnx_path) -> dict:
    onnxruntime.set_default_logger_severity(3)
    session = onnxruntime.InferenceSession(onnx_path)
    input_info: dict = {}
    for ip in session.get_inputs():
        name = ip.name
        shape = ip.shape
        if type(shape[0]) is not int:
            shape[0] = 1
        input_info.update({name: shape})
    return input_info

def get_input_shape_from_tflite(tflite_path)->dict:
    input_info: dict={}
    interpreter = tf.lite.Interpreter(tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_details = interpreter.get_input_details()
    for item in input_details:
       name = item["name"]
       n,c,h,w = item["shape"]
       input_info.update({name: [int(n), int(c), int(h), int(w)]})
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


def print_not_align_message(reason=None, is_tflite=False):
    logging.error("{}   Unfortunately   {}" .format("-" * 10, "-" * 10))
    if is_tflite == True:
       logging.error("The tflite model not aligned with tnn model\n")
    else:
       logging.error("The onnx model not aligned with tnn model\n")
    sys.exit(return_code.ALIGN_FAILED)


def print_align_message(is_tflite = False):
    logging.info("{}  Congratulations!   {}" .format("-" * 10, "-" * 10))
    if is_tflite == True:
       logging.info("The tflite model aligned with tnn model\n")
    else:
        logging.info("The onnx model aligned with tnn model\n")


def check_input_info(onnx_input_info: dict, tnn_input_info: dict):
    if len(onnx_input_info) != len(tnn_input_info):
        logging.info("input is not algin 186\n")
        print_not_align_message("onnx input size != tnn input size")
    for name, onnx_shape in onnx_input_info.items():
        tnn_name = convert_name.onnx_name2tnn_name(name)
        tnn_shape = tnn_input_info[tnn_name]
        if type(onnx_shape[0]) is not int:
            onnx_shape[0] = 1
        if tnn_shape != onnx_shape:
            logging.info("input is not algin 194\n")
            print_not_align_message(
                "The {}'s shape not equal! the onnx shape:{}, tnn shape: {}\n".format(name, str(onnx_shape),
                                                                                    str(tnn_shape)))

    logging.info("Check onnx input shape and tnn input shape align!\n")


def check_input_lite_info(onnx_input_info: dict, tnn_input_info: dict):
    if len(onnx_input_info) != len(tnn_input_info):
        print_not_align_message("tflite input size != tnn input size")
    for name, onnx_shape in onnx_input_info.items():
        tnn_name = convert_name.onnx_name2tnn_name(name)
        tnn_shape = tnn_input_info[tnn_name]
        if type(onnx_shape[0]) is not int:
            onnx_shape[0] = 1
        nchw = [1, 1, 1, 1]
        nchw[0] = onnx_shape[0]
        nchw[1] = onnx_shape[3]
        nchw[2] = onnx_shape[1]
        nchw[3] = onnx_shape[2]
        if tnn_shape != nchw:
            logging.info("input is not algin 216\n")
            print_not_align_message(
                "The {}'s shape not equal! the onnx shape:{}, tnn shape: {}\n".format(name, str(onnx_shape),
                                                                                      str(tnn_shape)))
    logging.info("Check tflite input shape and tnn input shape align!\n")

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
                refer_path: str = None, input_names: str = None, is_tflite: bool=False ) -> bool:
    """
    对 onnx 模型和 tnn 模型进行对齐.
    当前支持模型: 单输入,单输出;单输入,多输出;
    :param onnx_path:
    :param tnn_proto_path:
    :param tnn_model_path:
    :return:
    """
    logging.info("{}  align model (tflite or ONNX vs TNN),please wait a moment {}\n" .format("-" * 10, "-" * 10))

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
        if is_tflite == True:
            onnx_input_info = get_input_shape_from_tflite(onnx_path)
        else:
            onnx_input_info = get_input_shape_from_onnx(onnx_path)
    if is_tflite == True:
        check_input_lite_info(onnx_input_info, tnn_input_info)
    else:
       check_input_info(onnx_input_info, tnn_input_info)
    if input_file_path is None:
        # generate data
        input_path = data.gene_random_data(onnx_input_info)
    else:
        if os.path.exists(input_file_path):
            input_path = input_file_path
        else:
            logging.error("Invalid input_file_path")
            sys.exit(return_code.ALIGN_FAILED)
    if refer_path is None:
        if is_tflite == True:
            reference_output_path = run_tflite(onnx_path, input_path, onnx_input_info)
        else:
            reference_output_path = run_onnx(onnx_path, input_path, onnx_input_info)
    else:
        if os.path.exists(refer_path):
            reference_output_path = refer_path
        else:
            logging.error("Invalid refer_path")
            sys.exit(return_code.ALIGN_FAILED)
    run_tnn_model_check(tnn_proto_path, tnn_model_path, input_path, reference_output_path, is_tflite)
    if input_file_path is None and os.path.exists(input_path):
        data.clean_temp_data(os.path.dirname(input_path))
    if refer_path is None and os.path.exists(reference_output_path):
        data.clean_temp_data(reference_output_path)

    return True
