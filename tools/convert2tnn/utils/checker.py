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


import os
import onnxruntime
import re
import sys

from converter import logging
from utils import return_code


def check_file_exist(file_path):
    if os.path.exists(file_path) is False:
        logging.error("The " + file_path + " does not exist! please make sure the file exist!\n")
        sys.exit(return_code.CONVERT_FAILED)


def is_ssd_model(proto_path):
    proto_file = open(proto_path, 'r')
    lines = proto_file.read()
    proto_file.close()
    if "PriorBox" in lines:
        return True
    elif "DetectionOutput" in lines:
        return True
    else:
        return False

def check_onnx_dim(onnx_path : str):
    onnxruntime.set_default_logger_severity(3)
    so = onnxruntime.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1

    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'], sess_options=so)
    current_shape = {}
    status = 0
    for ip in session.get_inputs():
        current_shape[ip.name] = ip.shape
        for dim in ip.shape:
            if type(dim) is not int or dim < 0:
                status = -1
                
    if status == -1:
        return False, current_shape
    return True, current_shape
