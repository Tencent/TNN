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

from utils import args_parser
from onnx_converter import onnx2tnn
from caffe_converter import caffe2tnn
from tf_converter import tf2tnn
from utils import parse_path

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")


def main():
    args = args_parser.parse_args()
    if args.sub_command == 'onnx2tnn':
        onnx_path = parse_path.parse_path(args.onnx_path)
        output_dir = parse_path.parse_path(args.output_dir)
        input_names = args.input_names
        version = args.version
        optimize = args.optimize
        half = args.half
        align = args.align
        input_file = args.input_file_path
        ref_file = args.refer_file_path
        onnx_path = parse_path.parse_path(onnx_path)
        output_dir = parse_path.parse_path(output_dir)
        input_file = parse_path.parse_path(input_file)
        ref_file = parse_path.parse_path(ref_file)
        onnx2tnn.convert(onnx_path, output_dir, version, optimize, half, align, input_file, ref_file, input_names)
    
    elif args.sub_command == 'caffe2tnn':
        proto_path = parse_path.parse_path(args.proto_path)
        model_path = parse_path.parse_path(args.model_path)
        output_dir = parse_path.parse_path(args.output_dir)
        version = args.version
        optimize = args.optimize
        half = args.half
        align = args.align
        input_file = args.input_file_path
        ref_file = args.refer_file_path
        input_file = parse_path.parse_path(input_file)
        ref_file = parse_path.parse_path(ref_file)
        caffe2tnn.convert(proto_path, model_path, output_dir, version, optimize, half, align, input_file, ref_file)

    elif args.sub_command == 'tf2tnn':
        tf_path = parse_path.parse_path(args.tf_path)
        output_dir = parse_path.parse_path(args.output_dir)
        input_names = args.input_names
        output_names = args.output_names
        version = args.version
        optimize = args.optimize
        half = args.half
        align = args.align
        not_fold_const = args.not_fold_const
        input_file = args.input_file_path
        ref_file = args.refer_file_path
        input_file = parse_path.parse_path(input_file)
        ref_file = parse_path.parse_path(ref_file)
        tf2tnn.convert(tf_path, input_names, output_names, output_dir, version, optimize, half, align, not_fold_const, 
                        input_file, ref_file)
    else:
        print("Do not support convert!")


if __name__ == '__main__':
    main()
