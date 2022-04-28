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
import time


def parse_args():
    parser = argparse.ArgumentParser(prog='convert',
                                     description='convert ONNX/Tensorflow/Tensorflowlite/Caffe model to TNN model')
    subparsers = parser.add_subparsers(dest="sub_command")

    onnx2tnn_parser = subparsers.add_parser('onnx2tnn',
                                            help="convert onnx model to tnn model")
    onnx2tnn_parser.add_argument(dest='onnx_path',
                                 action='store',
                                 help="the path for onnx file")
    onnx2tnn_parser.add_argument('-in',
                                 metavar='input_info',
                                 dest='input_names',
                                 action='store',
                                 required=False,
                                 nargs='+',
                                 type=str,
                                 help="specify the input name and shape of the model. e.g., " +
                                      "-in input1_name:1,3,128,128 input2_name:1,3,256,256")
    onnx2tnn_parser.add_argument('-optimize',
                                 dest='optimize',
                                 default=False,
                                 action='store_true',
                                 required=False,
                                 help="If the model has fixed input shape, use this option to optimize the model for speed. On the other hand, if the model has dynamic input shape, dont use this option. It may cause warong result")
    onnx2tnn_parser.add_argument('-half',
                                 dest='half',
                                 default=False,
                                 action='store_true',
                                 required=False,
                                 help="save model using half")
    onnx2tnn_parser.add_argument('-v',
                                 metavar="v1.0.0",
                                 dest='version',
                                 default=time.strftime('%Y%m%d%H%M', time.localtime()),
                                 action='store',
                                 required=False,
                                 help="the version for model")
    onnx2tnn_parser.add_argument('-o',
                                 dest='output_dir',
                                 action='store',
                                 required=False,
                                 help="the output tnn directory")
    onnx2tnn_parser.add_argument('-align',
                                 dest='align',
                                 default='',
                                 action='store',
                                 required=False,
                                 choices=[None, 'output', 'all'],
                                 nargs='?',
                                 type=str,
                                 help='align the onnx model with tnn model. '
                                      'e.g., if you want to align the last output, you can use \'-align\' '
                                      'or \'-align output\'; '
                                      'if the model is not align, you can use \'-align all\' '
                                      'to address the first unaligned layer')
    onnx2tnn_parser.add_argument('-align_batch',
                                 dest='align_batch',
                                 default=False,
                                 action='store_true',
                                 required=False,
                                 help='align the onnx model with tnn model and check mutli batch')
    onnx2tnn_parser.add_argument('-input_file',
                                 dest='input_file_path',
                                 action='store',
                                 required=False,
                                 help="the input file path which contains the input data for the inference model.")
    onnx2tnn_parser.add_argument('-ref_file',
                                 dest='refer_file_path',
                                 action='store',
                                 required=False,
                                 help="the reference file path which contains the reference data to compare the results.")
    onnx2tnn_parser.add_argument('-debug',
                                 dest='debug',
                                 default=False,
                                 action='store_true',
                                 required=False,
                                 help="Turn on the switch to debug the model.")
    onnx2tnn_parser.add_argument('-int8',
                                 dest='int8',
                                 default=False,
                                 action='store_true',
                                 required=False,
                                 help="save model using dynamic range quantization. use int8 save, fp32 interpreting")

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
                                  default=time.strftime('%Y%m%d%H%M', time.localtime()),
                                  action='store',
                                  required=False,
                                  help="the version for model, default v1.0")
    caffe2tnn_parser.add_argument('-optimize',
                                  dest='optimize',
                                  default=False,
                                  action='store_true',
                                  required=False,
                                  help="If the model has fixed input shape, use this option to optimize the model for speed. On the other hand, if the model has dynamic input shape, dont use this option. It may cause warong result")
    caffe2tnn_parser.add_argument('-half',
                                  dest='half',
                                  default=False,
                                  action='store_true',
                                  required=False,
                                  help="save model using half")
    caffe2tnn_parser.add_argument('-align',
                                  dest='align',
                                  default='',
                                  action='store',
                                  required=False,
                                  choices=[None, 'output', 'all'],
                                  nargs='?',
                                  type=str,
                                  help='align the onnx model with tnn model. '
                                       'e.g., if you want to align the last output, you can use \'-align\' '
                                       'or \'-align output\'; '
                                       'if the model is not align, you can use \'-align all\' '
                                       'to address the first unaligned layer')
    caffe2tnn_parser.add_argument('-input_file',
                                  dest='input_file_path',
                                  action='store',
                                  required=False,
                                  help="the input file path which contains the input data for the inference model.")
    caffe2tnn_parser.add_argument('-ref_file',
                                  dest='refer_file_path',
                                  action='store',
                                  required=False,
                                  help="the reference file path which contains the reference data to compare the results.")
    caffe2tnn_parser.add_argument('-debug',
                                  dest='debug',
                                  default=False,
                                  action='store_true',
                                  required=False,
                                  help="Turn on the switch to debug the model.")
    caffe2tnn_parser.add_argument('-int8',
                                  dest='int8',
                                  default=False,
                                  action='store_true',
                                  required=False,
                                  help="save model using dynamic range quantization. use int8 save, fp32 interpreting")

    tf2tnn_parser = subparsers.add_parser('tf2tnn',
                                          help="convert tensorflow model to tnn model")
    tf2tnn_parser.add_argument('-tp',
                               dest="tf_path",
                               action='store',
                               required=True,
                               help="the path for tensorflow graphdef file")

    tf2tnn_parser.add_argument('-in',
                               metavar='input_info',
                               dest='input_names',
                               action='store',
                               nargs='+',
                               type=str,
                               required=True,
                               help="specify the input name and shape of the model. e.g., " +
                                    "-in input1_name:1,128,128,3 input2_name:1,256,256,3")
    tf2tnn_parser.add_argument('-on',
                               metavar='output_name',
                               dest='output_names',
                               action='store',
                               required=True,
                               nargs='+',
                               type=str,
                               help="the tensorflow model's output name. e.g. -on output_name1 output_name2")

    tf2tnn_parser.add_argument('-o',
                               dest='output_dir',
                               action='store',
                               required=False,
                               help="the output tnn directory")

    tf2tnn_parser.add_argument('-v',
                               metavar="v1.0",
                               dest='version',
                               default=time.strftime('%Y%m%d%H%M', time.localtime()),
                               action='store',
                               required=False,
                               help="the version for model")
    tf2tnn_parser.add_argument('-optimize',
                               dest='optimize',
                               default=False,
                               action='store_true',
                               required=False,
                               help="If the model has fixed input shape, use this option to optimize the model for speed. On the other hand, if the model has dynamic input shape, dont use this option. It may cause warong result")
    tf2tnn_parser.add_argument('-half',
                               dest='half',
                               default=False,
                               action='store_true',
                               required=False,
                               help="save the model using half")
    tf2tnn_parser.add_argument('-align',
                               dest='align',
                               default='',
                               action='store',
                               required=False,
                               choices=[None, 'output', 'all'],
                               nargs='?',
                               type=str,
                               help='align the onnx model with tnn model. '
                                    'e.g., if you want to align the last output, you can use \'-align\' '
                                    'or \'-align output\'; '
                                    'if the model is not align, you can use \'-align all\' '
                                    'to address the first unaligned layer')
    tf2tnn_parser.add_argument('-not_fold_const',
                               dest='not_fold_const',
                               default=False,
                               action='store_true',
                               required=False,
                               help=argparse.SUPPRESS)
    tf2tnn_parser.add_argument('-input_file',
                               dest='input_file_path',
                               action='store',
                               required=False,
                               help="the input file path which contains the input data for the inference model.")
    tf2tnn_parser.add_argument('-ref_file',
                               dest='refer_file_path',
                               action='store',
                               required=False,
                               help="the reference file path which contains the reference data to compare the results.")
    tf2tnn_parser.add_argument('-debug',
                               dest='debug',
                               default=False,
                               action='store_true',
                               required=False,
                               help="Turn on the switch to debug the model.")
    tf2tnn_parser.add_argument('-int8',
                               dest='int8',
                               default=False,
                               action='store_true',
                               required=False,
                               help="save model using dynamic range quantization. use int8 save, fp32 interpreting")

    # tflie parser
    tflite2tnn_parser = subparsers.add_parser('tflite2tnn',
                                              help="convert tensorflow-lite model to tnn model")
    tflite2tnn_parser.add_argument(dest="tf_path",
                                   action='store',
                                   help="the path for tensorflow-lite graphdef file")

    tflite2tnn_parser.add_argument('-o',
                                   dest='output_dir',
                                   action='store',
                                   required=False,
                                   help="the output tnn directory")

    tflite2tnn_parser.add_argument('-v',
                                   metavar="v1.0",
                                   dest='version',
                                   default=time.strftime('%Y%m%d%H%M', time.localtime()),
                                   action='store',
                                   required=False,
                                   help="the version for model")

    tflite2tnn_parser.add_argument('-half',
                                   dest='half',
                                   default=False,
                                   action='store_true',
                                   required=False,
                                   help="optimize the model")

    tflite2tnn_parser.add_argument('-align',
                                   dest='align',
                                   default='',
                                   action='store',
                                   required=False,
                                   help='align the onnx model with tnn model. '
                                        'e.g., if you want to align the final output, you can use -align output'
                                        'if you want to align whole model, you can use -align all')

    tflite2tnn_parser.add_argument('-input_file',
                                   dest='input_file_path',
                                   action='store',
                                   required=False,
                                   help="the input file path which contains the input data for the inference model.")

    tflite2tnn_parser.add_argument('-ref_file',
                                   dest='refer_file_path',
                                   action='store',
                                   required=False,
                                   help="the reference file path which contains the reference data to compare the results.")

    tflite2tnn_parser.add_argument('-debug',
                                   dest='debug',
                                   default=False,
                                   action='store_true',
                                   required=False,
                                   help="Turn on the switch to debug the model.")
    tflite2tnn_parser.add_argument('-int8',
                                   dest='int8',
                                   default=False,
                                   action='store_true',
                                   required=False,
                                   help="save model using dynamic range quantization. use int8 save, fp32 interpreting")

    return parser
