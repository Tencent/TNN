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

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='convert caffe model to onnx')

    parser.add_argument(dest='proto_file',
                        action='store',
                        help='the path for prototxt file, the file name must end with .prototxt')

    parser.add_argument(dest='caffe_model_file',
                        action='store',
                        help='the path for caffe model file, the file name must end with .caffemodel!')

    parser.add_argument('-o',
                        dest='onnx_file',
                        action='store',
                        help='the path for generate onnx file')
    args = parser.parse_args()
    return args
