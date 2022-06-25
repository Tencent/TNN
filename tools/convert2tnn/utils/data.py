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
import numpy as np
import os
import pathlib
from utils import cmd
from utils import checker

from converter import logging


def gene_random_data(input_info: dict) -> str:
    data = {}
    current_dir = pathlib.Path(__file__).parent.parent
    data_dir = os.path.join(current_dir, "temp_data")
    command = "mkdir -p " + data_dir
    
    logging.debug(command)

    cmd.run(command)
    checker.check_file_exist(data_dir)
    data_path = os.path.join(data_dir, "input.txt")
    data_file = open(data_path, "w")
    data_file.write(str(len(input_info)) + '\n')
    for name, info in input_info.items():
        shape = info['shape']
        data_type = info['data_type']
        data_file.write(name + ' ' + str(len(shape)) + ' ' + ' '.join([str(dim) for dim in shape]) + ' ' + str(data_type) + '\n')
        if data_type == 0:
            data[name] = np.random.rand(*shape)
            np.savetxt(data_file, data[name].reshape(-1), fmt="%0.6f")
        elif data_type == 2 or data_type == 3:
            # range [low, high)
            data[name] = np.random.randint(low=0, high=2, size=shape)
            np.savetxt(data_file, data[name].reshape(-1), fmt="%i")
    data_file.close()
    return data_path


def clean_temp_data(path: str):
    command = "rm -rf " + path
    cmd.run(command)
