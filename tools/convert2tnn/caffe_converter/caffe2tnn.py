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
import time


def caffe2onnx(proto_path, model_path, output_path):
    work_dir = "../caffe2onnx/"
    command = "python3 caffe2onnx.py " + proto_path + " " + model_path + " -o " + output_path
    result = cmd.run(command, work_dir=work_dir)
    if result == 0:
        return True
    else:
        return False


def convert(proto_path, model_path, output_dir, version, optimize, half, align,
            input_path=None, refer_path=None, debug_mode: bool = False):
    logging.info("Converter Caffe to ONNX Model\n")
    checker.check_file_exist(proto_path)
    checker.check_file_exist(model_path)
    if output_dir is None:
        output_dir = os.path.dirname(proto_path)
    checker.check_file_exist(output_dir)

    proto_name = os.path.basename(proto_path)
    proto_name = proto_name[:-len(".prototxt")]
    onnx_path = os.path.join(output_dir, proto_name + ".onnx")

    if caffe2onnx(proto_path, model_path, onnx_path) is False:
        logging.error("Oh No, caff2onnx failed :(\n")
        sys.exit(return_code.CONVERT_FAILED)
    else:
        logging.info("Congratulations! caffe2onnx succeed!\n")
    if version is None:
        version = time.strftime('%Y%m%d%H%M', time.localtime())

    is_ssd = checker.is_ssd_model(proto_path)
    if is_ssd:
        onnx2tnn.convert(onnx_path, output_dir, version, False, half, is_ssd=True)
    else:
        onnx2tnn.convert(onnx_path, output_dir, version, optimize, half)

    if is_ssd and ((input_path is None) or (refer_path is None)):
        align = False
        optimize = False

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
        is_opt = '.opt' if optimize else ''
        onnx_base_name = os.path.basename(onnx_path)
        is_align_all = (align == 'all')
        align_model.align_all(onnx_path, tnn_proto_path,
                              is_align_all, None, input_path, refer_path)
