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
from utils import align_model
from utils import return_code
from converter import logging

import os
import sys

def tflite2tnn(tf_path, tnn_path, not_fold_const=False):
    cmd.run("pwd")
    relative_path = "bin/TnnConverter"
    TnnConverter_path = parse_path.parse_path(relative_path)
    checker.check_file_exist(TnnConverter_path)
    command = TnnConverter_path + " -mt TFLITE  -mp " + tf_path
    checker.check_file_exist(TnnConverter_path)
    checker.check_file_exist(tf_path)
    if tnn_path is None:
        tnn_path = os.path.dirname(tf_path)
    checker.check_file_exist(tnn_path)
    command = command + " -od " + tnn_path + "/"
    logging.debug(command)
    result = cmd.run(command)
    if result == 0:
        return True
    else:
        return False


def convert(tf_path,  output_dir, version,  align=False,
            input_path=None, refer_path=None, debug_mode: bool = False):
    checker.check_file_exist(tf_path)
    model_name = os.path.basename(tf_path)
    if output_dir is None or not os.path.isdir(output_dir):
        output_dir = os.path.dirname(tf_path)
    checker.check_file_exist(output_dir)
    model_name = model_name[:-len(".tflite")]
    if tflite2tnn(tf_path, output_dir) is False:
        logging.error("Oh No, tflite2tnn failed :(\n")
        sys.exit(return_code.CONVERT_FAILED)
    else:
        logging.info("Convert TensorFlowLite to TNN model succeed!\n")

    if version is None:
        version = "v1.0"
    if align == 'output':
        proto_suffix = '.tnnproto'
        model_suffix = '.tnnmodel'
        tnn_proto_name = model_name + proto_suffix
        tnn_model_name = model_name + model_suffix
        tnn_proto_path = os.path.join(output_dir, tnn_proto_name)
        tnn_model_path = os.path.join(output_dir, tnn_model_name)
        align_model.align_model(tf_path, tnn_proto_path, tnn_model_path, input_path, refer_path, None, True,
                                debug_mode=debug_mode)
