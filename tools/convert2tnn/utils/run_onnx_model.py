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
import onnx
import onnxruntime
import os

import numpy as np

from collections import OrderedDict
from utils.run_src_model import BaseRunner


class OnnxRunner(BaseRunner):
    def get_src_model_input_information(self) -> dict:
        onnxruntime.set_default_logger_severity(3)
        session = onnxruntime.InferenceSession(self.src_model_path)
        input_info: dict = {}
        for ip in session.get_inputs():
            name = ip.name
            shape = ip.shape
            data_type = 0
            if ip.type == 'tensor(float)':
                data_type = 0
            elif ip.type == 'tensor(int64)' or ip.type == 'tensor(int32)':
                data_type = 3
            elif ip.type == 'tensor(bool)':
                data_type = 2
            else:
                logging.error("Do not support input date type")
            if type(shape[0]) is not int:
                shape[0] = 1
            shape_information = {'shape': shape,
                                 'data_type': data_type}
            input_info.update({name: shape_information})

        return input_info

    def modify_src_model_output(self) -> bool:
        onnx_model = onnx.load(self.src_model_path)

        output_list = []
        for output in onnx_model.graph.output:
            output_list.append(output.name)

        if self.align_all:
            for node in onnx_model.graph.node:
                for output in node.output:
                    if output in output_list:
                        continue
                    onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])

        model_name = self.src_model_path.split("/")[-1][:-5]
        model_name = model_name + ".hack.onnx"
        self.modify_model_path = os.path.join(self.dump_dir_path, model_name)
        onnx.save(onnx_model, self.modify_model_path)

        return True

    def inference(self) -> dict:
        session = onnxruntime.InferenceSession(self.modify_model_path)
        outputs = [x.name for x in session.get_outputs()]
        dump_data = OrderedDict(zip(outputs, session.run(outputs, self.input_data)))

        return dump_data

    def dump_single_output(self, output_name: str, output_data: np.ndarray, full_message: bool):
        output_name = output_name.replace("/", "_")
        output_name = output_name.replace(":", "_")
        output_path = os.path.join(self.dump_dir_path, output_name + ".txt")
        with open(output_path, "w") as f:
            if full_message == True:
                # number of output
                f.write("1\n")
                output_shape = output_data.shape
                data_type = 0
                if output_data.dtype == np.int64 or output_data.dtype == np.int32:
                    data_type = 3
                elif output_data.dtype == np.int8 or output_data.dtype == np.bool:
                    data_type = 2

                description = "{} {} ".format(output_name, len(output_shape))
                for dim in output_shape:
                    description += "{} ".format(dim)
                description += "{}".format(str(data_type))
                f.write(description + "\n")

            # keep the same as run_onnx in align_model.py
            if output_data.dtype == np.int64 or output_data.dtype == np.int32:
                np.savetxt(f, output_data.reshape(-1), fmt="%d")
            elif output_data.dtype == np.int8 or output_data.dtype == np.bool:
                np.savetxt(f, output_data.reshape(-1), fmt="%d")
            elif output_data.dtype == np.float32 or output_data.dtype == np.float64:
                np.savetxt(f, output_data.reshape(-1), fmt="%0.6f")
            else :
                print("ERROR: dump_single_output dont support data type: " + str(output_data.dtype))
                return False

        return True
