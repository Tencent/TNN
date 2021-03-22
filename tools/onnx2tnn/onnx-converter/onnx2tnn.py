#!/usr/bin/env python
import os
import argparse
import sys
import time
import traceback
# sys.path.append('./onnx-optimizer')
# from onnx_optimizer import onnx_optimizer

import onnx
# from onnx import version_converter

import onnx2tnn


def check_file_exist(file_path):
    if os.path.exists(file_path) is False:
        print("the " + file_path + " does not exist! please make sure the file exist!")
        exit(-1)


def parse_path(path: str):
    if path is None:
        return None
    if path.endswith(".onnx") is False:
        print("please make sure the onnx file path end with  \'.onnx\'")
        exit(-1)
    if path.startswith("/"):
        return path
    elif path.startswith("./"):
        return os.path.join(os.getcwd(), path[2:])
    elif path.startswith("../"):
        abs_path = os.getcwd() + "/" + path
        return abs_path
    else:
        return os.path.join(os.getcwd(), path)


def do_optimize(onnx_net_path, input_shape):
    try:
        import onnx2tnn.onnx_optimizer.onnx_optimizer as opt
    except ImportError:
        import onnx_optimizer.onnx_optimizer as opt
    else:
        print("\n\n t fail")
        os.system(sys.executable + " onnx_optimizer " + onnx_net_path)
        return

    import multiprocessing
    ctx = multiprocessing.get_context('spawn')

    p = ctx.Process(target=opt.onnx_optimizer, args=(onnx_net_path, input_shape))
    p.start()
    p.join()
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_model_path', help='Input ONNX model path')
    parser.add_argument('-version', help='Algorithm version string')
    parser.add_argument('-optimize', help='Optimize model befor convert, 1:default yes, 0:no')
    parser.add_argument('-half', help='Save model using half, 1:yes, 0:default no')
    parser.add_argument('-o',
                        dest='output_dir',
                        required=False,
                        action='store',
                        help='the output dir for tnn model')
    parser.add_argument('-input_shape', 
                        required=False, 
                        action='store',
                        nargs='+',
                        help='manually-set static input shape, useful when the input shape is dynamic')
    args = parser.parse_args()
    onnx_net_path = args.onnx_model_path
    algo_version = args.version
    algo_optimize = args.optimize
    model_half = args.half
    output_dir = args.output_dir
    input_shape = None
    if args.input_shape is not None:
        input_shape = ""
        for item in args.input_shape:
            input_shape += (item + " ")

    if onnx_net_path is None:
        print('Please make sure the onnx model path is correct!')
        exit(-1)
    onnx_net_path = parse_path(onnx_net_path)

    if algo_optimize == None:
        algo_optimize = '1'
    if model_half == None:
        model_half = '0'

    if algo_version == None:
        print('Please add model version with -version=xxxx')
        return
    if output_dir is None:
        output_dir = os.path.dirname(onnx_net_path)
    check_file_exist(onnx_net_path)
    check_file_exist(output_dir)
    onnx_net_opt_path = onnx_net_path[:-5] + '.opt.onnx'
    if algo_optimize == '0':
        onnx_net_opt_path = onnx_net_path

    if "convert" not in dir(onnx2tnn):
        print("\nYou should compile onnx2tnn first !!!")
        print("You can find more compilation details in <path-to-tnn>/doc/cn/user/convert.md")
        exit(-1)

    # original_net = onnx.load(onnx_net_path)
    # converted_model = version_converter.convert_version(original_net, 10)
    print('0.----onnx version:' + str(onnx.__version__))

    print('algo_optimize ' + algo_optimize)
    print('onnx_net_opt_path ' + onnx_net_opt_path)
    if algo_optimize != '0':
        print("1.----onnx_optimizer: " + onnx_net_path)
        do_optimize(onnx_net_path, input_shape)

    # os.access('/python/test.py',os.F_OK)
    print("2.----onnx2tnn: " + onnx_net_opt_path)
    file_time = time.strftime("%Y%m%d %H:%M:%S", time.localtime())
    status = 0

    try:
        if input_shape is None:
            input_shape = ""
        status = onnx2tnn.convert(onnx_net_opt_path, output_dir, algo_version, file_time, 0 if model_half == '0' else 1, 0, input_shape)
    except Exception as err:
        status = -1
        traceback.print_exc()

    if status != 0:
        exit(status)

    print("3.----onnx2tnn status: " + str(status))


if __name__ == '__main__':
    main()
