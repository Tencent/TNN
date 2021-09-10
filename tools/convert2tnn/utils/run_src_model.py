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

import linecache
import logging
import os
import pathlib
import sys

import numpy as np

from multiprocessing import Pool
from utils import cmd


TNN_MAGIC_NUMBER = 0x0FABC0002
TNN_MAGIC_NUMBER_V2 = 0x0FABC0004

np.random.seed(0)


def print_align_message(is_tflite: bool = False):
    logging.info("{}  Congratulations!   {}".format("-" * 10, "-" * 10))
    logging.info("The {} model is aligned with tnn model\n" .format("tflite" if is_tflite else "onnx"))


def print_not_align_message(is_tflite=False):
    logging.error("{}   Unfortunately   {}" .format("-" * 10, "-" * 10))
    logging.error("The {} model is not aligned with tnn model\n" .format("tflite" if is_tflite else "onnx"))


class BaseRunner:
    def __init__(self, src_model_path: str, tnn_proto_path: str, align_all: bool,
                 input_names: str = None, input_file_path: str = None, refer_file_path: str = None,
                 is_tflite: bool = False):
        self.src_model_path = src_model_path
        self.tnn_proto_path = tnn_proto_path
        self.align_all = align_all
        self.input_names = input_names
        self.input_file_path = input_file_path
        self.refer_file_path = refer_file_path
        self.is_tflite = is_tflite

        self.dump_dir_path = ""
        self.modify_model_path = ""
        self.input_data = {}
        self.dump_data = {}

    def get_dump_dir_path(self) -> str:
        convert2tnn_path = pathlib.Path(__file__).parent.parent
        data_dir = os.path.join(convert2tnn_path, "temp_data/")

        if os.path.exists(data_dir):
            command = "rm -rf {}".format(data_dir)
            cmd.run(command)

        command = "mkdir {}".format(data_dir)
        cmd.run(command)

        return data_dir

    def create_dump_dir(self):
        self.dump_dir_path = self.get_dump_dir_path()

    def get_src_model_input_information(self) -> dict:
        pass

    def get_tnn_model_input_information(self) -> dict:
        input_info: dict = {}
        magic_number = \
        linecache.getline(self.tnn_proto_path, 1).strip('\n').strip('\"').strip(',').strip(' ').split(" ")[-1]
        magic_number = int(magic_number)
        if magic_number == TNN_MAGIC_NUMBER:
            line = linecache.getline(self.tnn_proto_path, 2).strip('\n').strip('\"').strip(',')
            input_list = line.split(':')
            for tnn_input in input_list:
                name, n, c, h, w = tnn_input.strip(' ').split(' ')
                size = 4
                shape = [int(n), int(c), int(h), int(w)]
                data_type = 0
                input_shape_info = {'shape': shape, 'data_type': data_type}
                input_info.update({name: input_shape_info})
        elif magic_number == TNN_MAGIC_NUMBER_V2:
            line = linecache.getline(self.tnn_proto_path, 2).strip('\n').strip('\"').strip(',')
            input_list = line.split(':')
            for tnn_input in input_list:
                # information: name size shape1 shape2... data_type
                information = tnn_input.strip(' ').split(' ')
                name = information[0]
                size = int(information[1])
                data_type = int(information[-1])
                shape: list = list(map(int, information[2:-1]))
                input_shape_info = {'shape': shape, 'data_type': data_type}
                input_info.update({name: input_shape_info})
        else:
            logging.error("Unspport TNN model version\n")
            sys.exit(-1)

        return input_info

    def check_shape_information(self, src_model_input_information: dict, tnn_model_input_information: dict) -> bool:
        onnx_shape = src_model_input_information['shape']
        onnx_data_type = src_model_input_information['data_type']
        tnn_shape = tnn_model_input_information['shape']
        tnn_data_type = src_model_input_information['data_type']
        if type(onnx_shape[0]) is not int:
            onnx_shape[0] = 1
        # if tnn has valid shape and onnx model's is str, use tnn shape
        if len(onnx_shape) == len(tnn_shape):
            for i in range(len(onnx_shape)):
                os = onnx_shape[i]
                ts = tnn_shape[i]
                if isinstance(ts, int) and not isinstance(os, int):
                    onnx_shape[i] = ts
                    src_model_input_information['borrow_shape'] = True
        if onnx_data_type == tnn_data_type and onnx_shape == tnn_shape:
            return True
        else:
            return False

    def check_input_information(self, src_model_input_information: dict, tnn_model_input_information: dict) -> bool:
        if len(src_model_input_information) != len(tnn_model_input_information):
            logging.info("input is not align 186\n")
            # print_not_align_message("onnx input size != tnn input size")
        for name, onnx_info in src_model_input_information.items():
            tnn_name = name.replace(":", "_")
            tnn_info = tnn_model_input_information[tnn_name]
            if self.check_shape_information(onnx_info, tnn_info):
                logging.info(name + ": input shape of onnx and tnn is aligned!\n")
            else:
                logging.error("input is not align 194\n")
                # print_not_align_message(
                #     "The {}'s shape not equal! the onnx shape:{}, tnn shape: {}\n".format(name, str(onnx_info),
                #                                                                           str(tnn_info)))
        return True

    def modify_src_model_output(self) -> bool:
        pass

    def generate_input_data(self, input_information: dict, tnn_model_input_information: dict) -> bool:
        self.input_data = {}
        data_path = os.path.join(self.dump_dir_path, "input.txt")
        data_file = open(data_path, "w")
        data_file.write(str(len(input_information)) + '\n')
        for name, info in input_information.items():
            tnn_name = name.replace(":", "_")
            tnn_info = tnn_model_input_information[tnn_name]
            shape = info['shape']
            if "borrow_shape" in info and info["borrow_shape"]:
                shape = tnn_info['shape']
                logging.info("Using tnn shape:{} for input:{}".format(shape,name))
            data_type = info['data_type']
            data_file.write(tnn_name + ' ' + str(len(shape)) + ' ' + ' '.join([str(dim) for dim in shape]) + ' ' + str(
                data_type) + '\n')
            if data_type == 0:
                self.input_data[name] = np.random.rand(*shape).astype(np.float32)
                np.savetxt(data_file, self.input_data[name].reshape(-1), fmt="%0.6f")
            elif data_type == 3:
                # range [low, high)
                self.input_data[name] = np.random.randint(low=0, high=2, size=shape).astype(np.int64)
                np.savetxt(data_file, self.input_data[name].reshape(-1), fmt="%i")
        data_file.close()

        return True

    def inference(self) -> dict:
        pass

    def dump_single_output(self, output_name: str, output_data: np.ndarray, full_message: bool):
        pass

    def dump_all_output(self, dump_data: dict) -> bool:
        param_list = []
        for name, data in dump_data.items():
            param_list.append((name, data, True))
        with Pool(4) as p:
            p.starmap(self.dump_single_output, param_list)

    def run_model_check(self) -> bool:
        model_check_path = os.path.join(self.dump_dir_path[:-10], "bin/model_check")
        tnn_model_path = self.tnn_proto_path[:-9] + ".tnnmodel"
        input_path = os.path.join(self.dump_dir_path, "input.txt")
        command = "{} -p {} -m {} -i {} -a {} -d NAIVE".format(
            model_check_path, self.tnn_proto_path, tnn_model_path, input_path, self.dump_dir_path)
        logging.debug(command)

        return cmd.run(command, log_level="error")

    def remove_dump_file(self) -> bool:
        command = "rm -rf {}" .format(self.dump_dir_path)
        cmd.run(command)

    def run(self):
        self.create_dump_dir()

        src_model_input_information = self.get_src_model_input_information()
        tnn_model_input_information = self.get_tnn_model_input_information()

        self.check_input_information(src_model_input_information, tnn_model_input_information)

        self.modify_src_model_output()

        self.generate_input_data(src_model_input_information, tnn_model_input_information)

        dump_data = self.inference()

        self.dump_all_output(dump_data)

        status = self.run_model_check()
        if status == 0:
            self.remove_dump_file()
            print_align_message(self.is_tflite)
            return
        print_not_align_message(self.is_tflite)

        return
