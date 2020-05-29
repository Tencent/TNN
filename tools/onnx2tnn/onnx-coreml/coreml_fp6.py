import os
import argparse
import sys
import time
# sys.path.append('./onnx-optimizer')
# from onnx_optimizer import onnx_optimizer

# from onnx import version_converter
from onnx_coreml import convert
import coremltools
from coremltools.models import MLModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coreml_model_path', help='Input CoreML model path')
    args = parser.parse_args()
    cml_net_path = args.coreml_model_path

    cml_net_path_fp16 = cml_net_path+'_half.mlmodel'

    cml_model = coremltools.utils.load_spec(cml_net_path)
    cml_model_fp16 = coremltools.utils.convert_neural_network_spec_weights_to_fp16(cml_model)
    coremltools.utils.save_spec(cml_model_fp16, cml_net_path_fp16)
    # https://www.jianshu.com/p/4703bc425564

if __name__ == '__main__':
    main()
