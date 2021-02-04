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

import logging
import os
import pathlib

from utils import cmd


def convert_to_tnn_name(src_name: str) -> str:
    tnn_name = src_name.replace(":", "_")

    return tnn_name


def print_align_message(is_tflite: bool = False):
    logging.info("{}  Congratulations!   {}".format("-" * 10, "-" * 10))
    logging.info("The {} model is aligned with tnn model\n" .format("tflite" if is_tflite else "onnx"))


def print_not_align_message(is_tflite=False):
    logging.error("{}   Unfortunately   {}" .format("-" * 10, "-" * 10))
    logging.error("The {} model is not aligned with tnn model\n" .format("tflite" if is_tflite else "onnx"))


def get_dump_dir_path() -> str:
    convert2tnn_path = pathlib.Path(__file__).parent.parent.parent.parent
    data_dir = os.path.join(convert2tnn_path, "temp_data")

    if os.path.exists(data_dir):
        command = "rm -rf {}" .format(data_dir)
        cmd.run(command)

    command = "mkdir {}" .format(data_dir)
    cmd.run(command)

    return data_dir
