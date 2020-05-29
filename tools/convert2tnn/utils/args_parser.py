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
    parser = argparse.ArgumentParser(prog='convert',
                                     description='convert ONNX/Tensorflow/Caffe model to TNN model')

    subparsers = parser.add_subparsers(dest="sub_command")

    onnx2tnn_parser = subparsers.add_parser('onnx2tnn',
                                            help="convert onnx model to tnn model")
    onnx2tnn_parser.add_argument(dest='onnx_path',
                                 action='store',
                                 help="the path for onnx file")
    onnx2tnn_parser.add_argument('-optimize',
                                 dest='optimize',
                                 default=False,
                                 action='store_true',
                                 required=False,
                                 help="optimize the model")
    onnx2tnn_parser.add_argument('-half',
                                 dest='half',
                                 default=False,
                                 action='store_true',
                                 required=False,
                                 help="save model using half")
    onnx2tnn_parser.add_argument('-v',
                                 metavar="v1.0.0",
                                 dest='version',
                                 default="v1.0.0",
                                 action='store',
                                 required=False,
                                 help="the version for model")
    onnx2tnn_parser.add_argument('-o',
                                 dest='output_dir',
                                 action='store',
                                 required=False,
                                 help="the output tnn directory")

    # convert caff2onnx -pp proto_path -mp model_path -o
    caffe2tnn_parser = subparsers.add_parser('caffe2tnn',
                                             help="convert caffe model to tnn model")
    caffe2tnn_parser.add_argument(metavar='prototxt_file_path',
                                  dest='proto_path',
                                  action='store',
                                  help="the path for prototxt file")
    caffe2tnn_parser.add_argument(metavar='caffemodel_file_path',
                                  dest='model_path',
                                  action='store',
                                  help="the path for caffemodel file")
    caffe2tnn_parser.add_argument('-o',
                                  dest='output_dir',
                                  action='store',
                                  required=False,
                                  help="the output tnn directory")
    caffe2tnn_parser.add_argument('-v',
                                  metavar="v1.0",
                                  dest='version',
                                  default="v1.0.0",
                                  action='store',
                                  required=False,
                                  help="the version for model, default v1.0")
    caffe2tnn_parser.add_argument('-optimize',
                                  dest='optimize',
                                  default=False,
                                  action='store_true',
                                  required=False,
                                  help="optimize the model")
    caffe2tnn_parser.add_argument('-half',
                                  dest='half',
                                  default=False,
                                  action='store_true',
                                  required=False,
                                  help="save model using half")

    tf2tnn_parser = subparsers.add_parser('tf2tnn',
                                          help="convert tensorflow model to tnn model")
    tf2tnn_parser.add_argument('-tp',
                               dest="tf_path",
                               action='store',
                               required=True,
                               help="the path for tensorflow graphdef file")

    tf2tnn_parser.add_argument('-in',
                               metavar='input_name',
                               dest='input_names',
                               action='store',
                               required=True,
                               help="the tensorflow model's input names")

    tf2tnn_parser.add_argument('-on',
                               metavar='output_name',
                               dest='output_names',
                               action='store',
                               required=True,
                               help="the tensorflow model's output name")

    tf2tnn_parser.add_argument('-o',
                               dest='output_dir',
                               action='store',
                               required=False,
                               help="the output tnn directory")

    tf2tnn_parser.add_argument('-v',
                               metavar="v1.0",
                               dest='version',
                               default="v1.0.0",
                               action='store',
                               required=False,
                               help="the version for model")
    tf2tnn_parser.add_argument('-optimize',
                               dest='optimize',
                               default=False,
                               action='store_true',
                               required=False,
                               help="optimize the model")

    tf2tnn_parser.add_argument('-half',
                               dest='half',
                               default=False,
                               action='store_true',
                               required=False,
                               help="optimize the model")

    args = parser.parse_args()
    return args
