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
from utils import fix_tnn_output

from converter import logging

import os
import sys
import time


def hack_name(names: list):
    hacked_names = ""
    for name in names:
        if name.endswith(":0"):
            hacked_names = hacked_names + name + ","
        else:
            hacked_names = hacked_names + name + ":0,"
    return hacked_names[:-1]


def format_input(arguments: list) -> dict:
    format_input_info: dict = {}
    for item in arguments:
        position = item.rfind(':')
        name, dims = item[0:position], item[position+1:]
        if not name.endswith(':0'):
            name += ':0'
        dims = '[' + dims + ']'
        format_input_info.update({name: dims})
    return format_input_info


def tf2onnx(tf_path, input_names, output_name, onnx_path, not_fold_const=False):
    work_dir = "./"
    input_info: dict = format_input(input_names)
    input_info_str: str = ""
    input_nchw_names: str = ""
    for item in input_info.items():
        input_info_str += item[0] + item[1] + ","
        input_nchw_names += item[0] + ","
    command = "python3 -m tf2onnx.convert  --graphdef " + tf_path
    command = command + " --inputs " + input_info_str
    command = command + " --inputs-as-nchw " + input_nchw_names

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
            input_path=None, refer_path=None, debug: bool = False, debug_mode: bool = False):
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
        version = time.strftime('%Y%m%d%H%M', time.localtime())
    checker.check_file_exist(onnx_path)
    onnx2tnn.convert(onnx_path, output_dir, version, optimize, half)

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

    if align == 'output':
        align_model.align_model(onnx_path, tnn_proto_path, tnn_model_path, input_path, refer_path, debug_mode=debug_mode)
    elif align == 'all':
        is_align_all = (align == 'all')
        align_model.align_all(onnx_path, tnn_proto_path,
                              is_align_all, input_names, input_path, refer_path)

    onnx_base_name = os.path.basename(onnx_path)
    tnn_proto_name = onnx_base_name[:-len('.onnx')] + ('.opt.tnnproto' if optimize else ".tnnproto")
    tnn_proto_path = os.path.join(output_dir, tnn_proto_name)

    fix_tnn_output.fix_tnn_output(tnn_proto_path)
