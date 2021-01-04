import argparse
import onnx_optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_model', help='Input ONNX model')
    parser.add_argument('-input_shape', type=str, nargs='+')
    args = parser.parse_args()
    onnx_net_path = args.onnx_model

    input_shape = None
    if args.input_shape is not None:
        input_shape = ""
        for item in args.input_shape:
            input_shape += (item + " ")
    onnx_optimizer.onnx_optimizer(onnx_net_path, input_shape)


if __name__ == '__main__':
    main()
