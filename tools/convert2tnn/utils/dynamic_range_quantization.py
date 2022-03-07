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

from converter import logging
from utils import checker
from utils import parse_path
from utils import cmd


def quantization(src_model_path: str, model_type: str, output_dir: str, optimize: bool) -> None:
    src_model_name = os.path.basename(src_model_path)

    support_model_type = ("prptotxt", "onnx", "pb", "tflite")
    if model_type not in support_model_type:
        logging.error("{} is not support in dynamic range quantization".format(model_type))
        return

    if output_dir is None:
        output_dir = os.path.dirname(src_model_path)

    model_name = src_model_name[:-len("." + model_type)]
    if optimize:
        model_name += ".opt"

    tnnproto = os.path.join(output_dir, model_name + ".tnnproto")
    tnnmodel = os.path.join(output_dir, model_name + ".tnnmodel")

    relative_path = "bin/dynamic_range_quantization"
    dynamic_range_quantization_path = parse_path.parse_path(relative_path)
    checker.check_file_exist(dynamic_range_quantization_path)
    command = "{} -p {} -m {} -qp {} -qm {}".format(dynamic_range_quantization_path, tnnproto, tnnmodel, tnnproto, tnnmodel)

    logging.debug("dynamic range quantization command: " + command)

    cmd.run(command)

    return
