import os
import argparse
import sys
import time
# sys.path.append('./onnx-optimizer')
# from onnx_optimizer import onnx_optimizer

import onnx
from onnx import helper, shape_inference
# from onnx import version_converter
import coremltools
from coremltools.models import MLModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_model_path', help='Input ONNX model full path')
    parser.add_argument('-mlmodelc', help='Save model wiht mlmodelc, 1:default yes, 0: no')
    args = parser.parse_args()

    onnx_net_path = args.onnx_model_path
    mlmodelc = args.mlmodelc
    if mlmodelc == None:
        mlmodelc = '1'

    onnx_net_dir = onnx_net_path[0:onnx_net_path.rfind('.')];
    print("dir: "+onnx_net_dir)

    cml_net_path = onnx_net_path+'.mlmodel'
    if onnx_net_path.endswith('.onnx'):
        cml_net_path = onnx_net_path[0:len(onnx_net_path)-5]+'.mlmodel'

    onnx_model = onnx.load(onnx_net_path)
    onnx_inputs = onnx_model.graph.input
    onnx_outputs = onnx_model.graph.output
    # #
    # onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 320
    # onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 320
    # onnx.checker.check_model(onnx_model)
    # onnx_model = shape_inference.infer_shapes(onnx_model)

    cml_model = coremltools.converters.onnx.convert(model=onnx_net_path, minimum_ios_deployment_target='13')
    # cml_model = convert(onnx_model, image_input_names=[onnx_inputs[0].name])
    cml_model.save(cml_net_path)
    # new_cml_spec = cml_model.get_spec()
    # coremltools.utils.save_spec(new_cml_spec, cml_net_path)

    # print("dir: "+onnx_net_dir)
    if mlmodelc == '1':
        cmd = '/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc compile '+cml_net_path+' '+onnx_net_dir
        os.system(cmd)

    # coremltools.utils.rename_feature(new_cml_spec, onnx_inputs[0].name, 'input_image')
    # coremltools.utils.rename_feature(new_cml_spec, onnx_outputs[0].name, 'output_array')
    # new_cml_spec.neuralNetwork.preprocessing[0].featureName = 'input_image'
    # # https://github.com/apple/coremltools/issues/244
    # coremltools.utils.save_spec(new_cml_spec, cml_net_path)
    # cml_model.save(cml_net_path)

if __name__ == '__main__':
    main()
