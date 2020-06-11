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
import cv2
import numpy as np
import os
from utils import cmd
from utils import checker


def gene_data_from_png(inputs, shape):
    # inputs format: input1:path1,input2,path2
    input_list = inputs.split(",")
    input_feed = {}
    for item in input_list:
        name, path = item.split(":")
        img = cv2.imread(path)
        img = img.astype(np.float32)
        img = cv2.resize(img, shape)
        img = img.transpose((2, 0, 1))
        current_shape = img.shape
        img = img.reshape(1, *current_shape)
        input_feed[name] = img

    return input_feed


def gene_random_data(input_info: dict) -> str:
    data = {}
    # data_dir = os.path.join(os.getcwd(), "./data_dir")
    data_dir = "./data_dir"
    command = "mkdir -p " + data_dir
    print(command)
    cmd.run("pwd")
    cmd.run(command)
    checker.check_file_exist(data_dir)
    data_path = os.path.join(data_dir, "input.txt")
    data_file = open(data_path, "w")
    for name, shape in input_info.items():
        data[name] = np.random.rand(*shape)
        np.savetxt(data_file, data[name].reshape(-1), fmt="%0.18f")
    data_file.close()
    return data_path

    
def remove_temp_random_data():
    cmd.run("rm -rf ./data_dir")

