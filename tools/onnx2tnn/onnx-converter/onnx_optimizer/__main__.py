import argparse
import onnx_optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_model', help='Input ONNX model')
    args = parser.parse_args()
    onnx_net_path = args.onnx_model
    onnx_optimizer.onnx_optimizer(onnx_net_path)


if __name__ == '__main__':
    main()
