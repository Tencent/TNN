import os

import argparse
# import pickle

import onnx
import onnx.utils
# from onnx import optimizer

# import onnxsim #pip3 install onnx-simplifier
# import sys
# sys.path.append('./onnx_simplifier/onnxsim')
from onnxsim import onnx_simplifier

def onnx_fix_prelu(m: onnx.ModelProto) -> None:
    tensor_name_to_fix = set()

    for node in m.graph.node:
        if node.op_type == 'PRelu':
            tensor_name_to_fix.add(node.input[1])  # 2nd input is the weight

    for init in m.graph.initializer:
        if init.name not in tensor_name_to_fix:
            continue
        if len(init.dims) > 1:
            print('%s, PRelu to fix expect weight rank == 1' % init.name, file=sys.stderr)
            continue
        # We only support NCHW layput
        c = init.dims.pop()
        init.dims.extend([1, c, 1, 1])

    for inp in m.graph.input:
        if inp.name not in tensor_name_to_fix:
            continue
        if len(inp.type.tensor_type.shape.dim) > 1:
            print('%s, PRelu to fix expect weight rank == 1' % inp.name, file=sys.stderr)
            continue
        # We only support NCHW layput
        c = inp.type.tensor_type.shape.dim.pop().dim_value
        d1 = onnx.TensorShapeProto.Dimension()
        d2 = onnx.TensorShapeProto.Dimension()
        d3 = onnx.TensorShapeProto.Dimension()
        d4 = onnx.TensorShapeProto.Dimension()
        d1.dim_value = 1
        d2.dim_value = c
        d3.dim_value = 1
        d4.dim_value = 1
        inp.type.tensor_type.shape.dim.extend([d1, d2, d3, d4])

def onnx_optimizer(onnx_net_path, input_shape=None):
        onnx_net_opt_path = onnx_net_path[:-5]+'.opt.onnx'

        print(os.getcwd())

        print("----load onnx model: "+onnx_net_path)
        onnx_model = onnx.load(onnx_net_path)

        # all_passes = optimizer.get_available_passes()
        # print("----available optimization passes:")
        # for p in all_passes:
        #     print(p)
        # print()
        #
        # print("----optimize onnx model: "+onnx_net_path)
        # passes = ['eliminate_nop_pad',
        #           'eliminate_identity',
        #           'extract_constant_to_initializer',
        #           'fuse_bn_into_conv',
        #           'fuse_add_bias_into_conv',
        #           'fuse_pad_into_conv',
        #           'fuse_matmul_add_bias_into_gemm']
        passes = ['eliminate_identity',
                  'eliminate_nop_dropout',
                  'eliminate_nop_monotone_argmax',
                  'eliminate_nop_pad',
                  'eliminate_nop_transpose',
                  'extract_constant_to_initializer',
                  'fuse_bn_into_conv',
                  'fuse_add_bias_into_conv',
                  'fuse_consecutive_concats',
                  'fuse_consecutive_log_softmax',
                  'fuse_consecutive_reduce_unsqueeze',
                  'fuse_consecutive_squeezes',
                  'fuse_consecutive_transposes',
                  'fuse_matmul_add_bias_into_gemm',
                  'fuse_pad_into_conv',
                  'fuse_transpose_into_gemm']

        #try:
        #    optimized_onnx_model = optimizer.optimize(onnx_model, passes)
        #    optimized_onnx_model = onnx.utils.polish_model(optimized_onnx_model)
        #except IndexError as e:
        #    optimized_onnx_model = onnx_model
        optimized_onnx_model = onnx_model

        try:
            input_shapes_ = {}
            if (input_shape is not None) and (input_shape != ""):
                input_shape = input_shape.strip()
                for x in input_shape.split(" "):
                    if ':' not in x:
                        input_shapes_[None] = list(map(int, x.split(',')))
                    else:
                        pieces = x.split(':')
                        # for the input name like input:0
                        name, shape = ':'.join(
                            pieces[:-1]), list(map(int, pieces[-1].split(',')))
                        input_shapes_[name] = shape
            optimized_onnx_model, check_ok = onnx_simplifier.simplify(
                optimized_onnx_model, input_shapes=input_shapes_, perform_optimization=False)
            if not check_ok :
                print("Check failed!")
                exit()
        except IndexError as e:
            print("----onnxsim.simplify error: You'd better check the result with Netron")
            print("----onnxsim.simplify error: "+str(RuntimeError))
        except RuntimeError:
            print("----onnxsim.simplify error: You'd better check the result with Netron")
            print("----onnxsim.simplify error: "+str(RuntimeError))

        # optimized_onnx_model = onnx_model
        print("----export optimized onnx model: "+onnx_net_opt_path)
        onnx.save(optimized_onnx_model, onnx_net_opt_path)
        print("----export optimized onnx model done")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('onnx_model', help='Input ONNX model')
#     args = parser.parse_args()
#     onnx_net_path = args.onnx_model
#     onnx_optimizer(onnx_net_path)
#
# if __name__ == '__main__':
#     main()
