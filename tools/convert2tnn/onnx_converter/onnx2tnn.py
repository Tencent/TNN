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
from utils import align_model

from converter import logging

import os
import sys


def throw_exception(current_shape):
    message = "Current shape: "
    for name, shape in current_shape.items():
        message += str(name) + ": " + str(shape) + "   "

    logging.error("You should use -in to specify input's name and shape. e.g.: -in name:1,3,32,32")
    logging.error(message)

    sys.exit(return_code.CONVERT_FAILED)


def check_input_names(input_names: str, onnx_input_info: dict):
    input_names = input_names.strip()

    input_shapes_ = {}
    for x in input_names.split(" "):
        if ':' not in x:
            input_shapes_[None] = list(map(int, x.split(',')))
        else:
            pieces = x.split(':')
            # for the input name like input:0
            name, shape = ':'.join(
                pieces[:-1]), list(map(int, pieces[-1].split(',')))
            input_shapes_[name] = shape

    if len(input_shapes_) != len(onnx_input_info):
        logging.error("The specified input does not match the input of the ONNX model.")
        logging.error("Specified: ", list(input_shapes_.keys()))
        logging.error("ONNX: ", list(onnx_input_info.keys()))
        sys.exit(return_code.CONVERT_FAILED)


def convert(onnx_path, output_dir=None, version="v1.0", optimize=True, half=False, align='', align_batch=False,
            input_path=None, refer_path=None, input_names: str = None, is_ssd=False, debug_mode: bool = False):
    """
    执行 onnx 转换为 tnn 的转换指令
    :parameter:
          onnx_path:    需要转换的 onnx 文件的路径
          output_path:  生成的 tnn 文件的路径
          version:      转换模型的版本号
          optimize:     是否需要对模型进行优化,默认是需要进行优化
          half:         是否需要转为 FP16 的模型,减小模型的大小
    :return return_code
    :exception 执行超时
    """
    logging.info("Converter ONNX to TNN Model...\n")

    checker.check_file_exist(onnx_path)

    try:
        if not is_ssd:
            logging.info("Converter ONNX to TNN check_onnx_dim...\n")
            ret, current_shape = checker.check_onnx_dim(onnx_path)
            logging.info("Converter ONNX to TNN check_onnx_dim...\n")
            if ret is False and current_shape is not None:
                if input_names is None:
                    logging.info("Converter ONNX to TNN current_shape...\n")
                    throw_exception(current_shape)
            if input_names is not None:
                input_names = input_names.strip()
                if ":" not in input_names and " " not in input_names:
                    input_names = list(current_shape.keys())[0] + ":" + input_names
                check_input_names(input_names, current_shape)
    except Exception as e:
        print(e)
        logging.error("check_onnx_dim failed, next stage of convertion may failed too\n")
        
    proto_suffix = '.tnnproto'
    model_suffix = '.tnnmodel'
    command = "python3 onnx2tnn.py " + onnx_path
    command = command + " -version=" + version
    checker.check_file_exist(onnx_path)
    if optimize is True:
        command = command + " -optimize=1"
    else:
        command = command + " -optimize=0"
    if half is True:
        command = command + " -half=1"
    else:
        command = command + " -half=0"

    if output_dir is None:
        output_dir = os.path.dirname(onnx_path)
    checker.check_file_exist(output_dir)
    command = command + " -o " + output_dir

    if input_names is not None:
        command = command + " -input_shape " + input_names
    logging.debug("The onnx2tnn command:" + command + "\n")

    current_file_dir = os.path.dirname(__file__)

    work_dir = current_file_dir + "/../../onnx2tnn/onnx-converter/"
    result = cmd.run(command, work_dir=work_dir)

    if result == 0:
        logging.info("Converter ONNX to TNN model succeed!\n")
    else:
        logging.error("Converter ONNX to TNN model failed!\n")
        sys.exit(return_code.CONVERT_FAILED)
    onnx_base_name = os.path.basename(onnx_path)

    if optimize is True:
        tnn_proto_name = onnx_base_name[:-len('.onnx')] + '.opt' + proto_suffix
        tnn_model_name = onnx_base_name[:-len('.onnx')] + '.opt' + model_suffix
    else:
        tnn_proto_name = onnx_base_name[:-len('.onnx')] + proto_suffix
        tnn_model_name = onnx_base_name[:-len('.onnx')] + model_suffix
    tnn_proto_path = os.path.join(output_dir, tnn_proto_name)
    tnn_model_path = os.path.join(output_dir, tnn_model_name)

    if align == 'output' or align_batch is True:
        if input_names is None:
            align_model.align_model(onnx_path, tnn_proto_path, tnn_model_path, input_path, refer_path,
                                    debug_mode=debug_mode, align_batch=align_batch)
        else:
            align_model.align_model(onnx_path, tnn_proto_path, tnn_model_path, input_path, refer_path, input_names,
                                    debug_mode=debug_mode, align_batch=align_batch)
    elif align == 'all':
        is_align_all = (align == 'all')
        align_model.align_all(onnx_path, tnn_proto_path,
                                          is_align_all, input_names, input_path, refer_path)
