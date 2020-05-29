# Model Conversion Introduction

## Overview

<div align=left><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/user/resource/convert.png"/>

TNN currently supports the industry's mainstream model file formats, including ONNX, Pytorch, Tensorflow and Caffe. As shown in the figure above, TNN utilises ONNX as the port to support multiple model file formats. 
If you want to convert model file formats such as Pytorch, Tensorflow, and Caffe to TNN, you need to use corresponding model conversion tool to convert the original format into ONNX model first, which then will be transferred into a TNN model.

| Source Model   | Convertor        | Target Model |
|------------|-----------------|----------|
| Pytorch    | pytorch export directly | ONNX     |
| Tensorflow | tensorflow-onnx | ONNX     |
| Caffe      | caffe2onnx      | ONNX     |
| ONNX       | onnx2tnn        | TNN      |

At present, TNN only supports common network structures such as CNN. Networks like RNN and GAN are under development.

## I. Convert ONNX to TNN model
We provide convert2tnn, an integrated conversion tool which can convert Tensorflow, Caffe and ONNX models into TNN models. Since 
Pytorch is able to directly export ONNX models, this tool no longer provides model conversion for Pytorch.

### Use onnx2tnn in Mac

#### Environment prerequisite

1. install protobuf(version >= 3.7.1)  
  ```shell script
  brew install protobuf
  brew link --overwrite protobuf
  ```
note: Inconsistent versions between protobuf and coverter might cause errors, it is best to keep the same version between two. Otherwise use the latest version, and recompile the convertor lib.

2. install python (version >=3.6)  
```shell script
brew install python3
```

3. install onnx、onnxruntime、numpy（onnx=1.6.0 onnxruntime=1.1.0 numpy>=1.17.0） 
```shell script
pip3 install onnx onnxruntime numpy
```

#### onnx2tnn Compilation

 Build script is provided on Mac, run the commands below:
 ```shell script
cd <path-to-tnn>/tools/onnx2tnn/onnx-converter
./build.sh 
 ```
#### onnx2tnn Usage

```shell script
python3 onnx2tnn.py model.onnx -version=algo_version -optimize=1 -half=0
```
```text
parameter description：
-version
the version of the model，facilitate the tracking of subsequent flows

-optimize
1(default): It is used to do fusion operators in the fusion, e.g. Fuse BN + Scale into the Conv layer
0 ：If the fusion fails, you can try to set this value

-half
1: Store model in FP16，reduce model size。
0（default）: Store model as FP32。
Note: Whether the actual calculation uses FP16 or not depends on the target platform. The mobile GPU currently only supports FP16 calculations.
```

### Use onnx2tnn in Linux

#### Environment prerequisite  

 The following configuration is for Centos 7.2. Ubuntu system can also be applied with modified installation commands.
1. install protobuf （version >= 3.4.0）

Download the latest version of protobuf directly from the official website, and then install it according to the documentation.  

2. install pythinstallon3 and python-devel（version >= 3.6. 8）  

```shell script
yum install python3 python3-devel
```

3. install dependencies for python

```shell script
pip3 install pytest numpy onnx onnxruntime
```

4. cmake （version >= 3.0）
Download the latest version of cmake from the official website and install it according to the documentation. It is recommended to use the latest version of cmake.

#### build

1. swtich to the coverter dir
```shell script
cd <path-to-tnn>/tools/onnx2tnn/onnx-converter
```

2. build pybind11 

```shell script
cd <path_to_tnn_root>/tools/onnx2tnn/onnx-converter/pybind11/
mkdir build
cd build
cmake ..
make check -j 4
```

3. build onnx_converter
```shell script
# create new dir for build
mkdir build
cd build cmake ./../
make -j 4

#copy the generated lib
cp ./*.so ./../

#delete the build dir
cd ./../
rm -r build
```

#### onnx2tnn Usage

```shell script
python3 onnx2tnn.py model.onnx -version=algo_version -optimize=1 -half=0
```
```text
parameter description：
-version
the version of the model，facilitate the tracking of subsequent flows

-optimize
1(default): It is used to do fusion operators in the fusion, e.g. Fuse BN + Scale into the Conv layer
0 ：If the fusion fails, you can try to set this value

-half
1: Store model in FP16，reduce model size。
0（default）: Store model as FP32。
Mote: Whether the actual calculation using FP16 or not depends on the target platform. The mobile GPU currently only supports FP16 calculations.
```

### supported Ops in onnx2tnn

| onnx                                                         | tnn                             |
|--------------------------------------------------------------|---------------------------------|
| AveragePool                                                  | Pooling / Pooling3D             |
| BatchNormalization                                           | BatchNormCxx                    |
| Clip                                                         | ReLU6                           |
| Concat                                                       | Concat                          |
| Conv                                                         | Convolution3D / Convolution     |
| ConvTranspose(ConvTranspose+BatchNormalization)              | Deconvolution3D / Deconvolution |
| DepthToSpace                                                 | Reorg                           |
| Div                                                          | Mul                             |
| Gemm                                                         | InnerProduct                    |
| GlobalAveragePool                                            | Pooling / Pooling3D             |
| GlobalMaxPool                                                | Pooling / Pooling3D             |
| InstanceNormalization                                        | InstBatchNormCxx                |
| LeakyRelu                                                    | PReLU                           |
| MaxPool                                                      | Pooling / Pooling3D             |
| Mul                                                          | Mul                             |
| Normalize(ReduceL2 + Clip+ Expand+Div)                       | Normalize                       |
| PReLU                                                        | PReLU                           |
| Pad                                                          | Pad / Pad3D                     |
| ReduceMean                                                   | ReduceMean                      |
| Relu                                                         | ReLU                            |
| Reshape                                                      | Reshape                         |
| ShuffleChannle(Reshape+Transpose+Reshape)                    | ShuffleChannle                  |
| Slice                                                        | StridedSlice                    |
| Softmax(Exp + ReduceSum + Div)                               | SoftmaxCaffe                    |
| Softmax(Transpose + Reshape + Softmax + Reshape + Transpose) | SoftmaxCaffe                    |
| Softplus                                                     | Softplus                        |
| Split                                                        | SplitV                          |
| Sub                                                          | BatchNormCxx                    |
| Tanh                                                         | TanH                            |
| Tile                                                         | Repeat                          |
| Transpose                                                    | Transpose                       |
| Upsample                                                     | Upsample                        |



### *Limitations*
1. Currently tnn2onnx only supports 4-dimensional (nchw) data types.
2. It is recommended to set the batch size of the model input to 1, but not a relatively large value.
3. Currently, the asymmetric padding in pooling layer is not supported. (This special situation will appear in "pool5 / 7x7 \ _s1 \ _1" in the inceptionv1 model)

## II. Convert Pytorch model to ONNX model
Pytorch supports direct conversion of the trained model to the ONNX model, so using the Pytorch's own export method will easily export the Pytorch model to ONNX one. The following code shows how to export the resnet50 Pytorch model to the ONNX model.
Pytorch also provides a more detailed documentation on how to export the Pytorch model as an ONNX model. For more information, please refer to the following link:
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

```python
import torch.hub
import numpy as np

torch.hub.list('pytorch/vision')

se_resnet50 = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,)

senet = se_resnet50()
senet.load_state_dict(torch.load("./seresnet50-60a8950a85b2b.pkl"))
senet.eval()
random_data = np.random.rand(1, 3, 224, 224).astype(np.float)
torch.onnx.export(senet,
				  random_data,
				  "./sent.onnx",
				  export_params=True,
				  opset_version=11,
				  do_constant_folding=True,
				  input_names= ['input'],
				  output_names = ['output'])
```


## III. Convert Tensorflow model to ONNX model
The tf2onnx tool is an open source conversion tool provided by ONNX, which can directly convert Tensorflow models to ONNX models.

### tf2onnx install
``` shell script
pip3 install -U tensorflow onnxruntime tf2onnx

```
Note: It is recommended to install tensorflow version 1.15.0. Actually, this tool currently does not support TensorFlow 2.0 very well.

### tf2onnx usage

The following is the command to convert test.pb, which is very convenient to use. It is recommended that you read tf2onnx's README.md, which contains detailed descriptions of the various parameters of the tool.
``` shell script
python3 -m tf2onnx.convert  --graphdef test.pb   --inputs input_data:0  --outputs pred:0   --opset 11 --output  test.onnx --inputs-as-nchw input_data:0
```

project link：https://github.com/onnx/tensorflow-onnx


## IV. Convert Caffe models to ONNX models

caffe2onnx tool is designed to convert the caffe model into the onnx model, which then utilises onnx2tnn tool to convert the onnx model into a TNN model.
project link： '<tnn_root_path>/tools/caffe2onnx'

### Environment prerequisite(Mac and Linux)  
1. protobuf(version >= 3.4.0)  
Install the latest version from the official website according to the documentation. If it is a Mac system, use brew to install it directly.
```shell script
brew install protobuf
```

2. python( version >= 3.6)
```shell script
brew install python
```

3. onnx(1.6.0)
```shell script
pip3 install onnx numpy
```


### caffe2onnx usage
1. download
``` shell script
cd  <tnn_root_path>/tools/caffe2onnx
```
2. coversion between different caffe versions

At present, the caffe2onnx tool currently only supports the latest version of the caffe format, so you need to convert the old version of the caffe network and model to the new version before using the caffe2onnx tool. Caffe comes with tools to convert the old version model to the new version model. The specific usage is as follows:
```shell script
upgrade_net_proto_text [old prototxt] [new prototxt]
upgrade_net_proto_binary [old caffemodel] [new caffemodel]
```
The format of the modified input is as follows:

```text
layer {
  name: "data"
  type: "input"
  top: "data"
  input_param { shape: { dim: 1 dim: 3 dim: 224 dim: 224 } }
}
```
3. caffe2onnx usage

```shell script
python3 convert2onnx.py ./test.prototxt ./test.caffemodel -o ./test.onnx
```

```text
usage: convert2onnx.py [-h] [-o ONNX_FILE] proto_file caffe_model_file

convert caffe model to onnx

positional arguments:
  proto_file        the path for prototxt file, the file name must end with
                    .prototxt
  caffe_model_file  the path for caffe model file, the file name must end with
                    .caffemodel!

optional arguments:
  -h, --help        show this help message and exit
  -o ONNX_FILE      the path for generate onnx file
```

### supported operators by caffe2onnx

| Number | caffe layer            | onnx operator                                         |
| ------ | ---------------------- | ----------------------------------------------------- |
| 1      | BatchNorm              | BatchNormalization                                    |
| 2      | BatchNorm + Scale      | BatchNormalization                                    |
| 3      | Concat                 | Concat                                                |
| 4      | Convolution            | Conv                                                  |
| 5      | ConvolutionDepthwise   | Conv                                                  |
| 6      | Deconvolution          | ConvTranspose                                         |
| 7      | DetectionOutput        | DetectionOutput(customer defination)                  |
| 8      | Dropout                | Dropout                                               |
| 9      | Eltwise                | Mul/Add/Max                                           |
| 10     | Flatten                | Reshape                                               |
| 11     | InnerProduct           | Reshape + Gemm                                        |
| 12     | LRN                    | LRN                                                   |
| 13     | MaxUnPool              | MaxUnPool                                             |
| 14     | PReLU                  | PRelu                                                 |
| 15     | Permute                | Transpose                                             |
| 16     | Pooling                | MaxPool/AveragePool/GlobalMaxPool/GlobalAveragePool   |
| 17     | PriorBox               | PriorBox(customer defination)                         |
| 18     | ReLU                   | Relu/LeakyRelu                                        |
| 19     | ReLU6                  | Clip                                                  |
| 20     | Reshape                | Reshape                                               |
| 21     | Scale                  | Mul + Reshape                                         |
| 22     | ShuffleChannel         | Reshape + Transpose + Reshape                         |
| 23     | Sigmoid                | Sigmoid                                               |
| 24     | Slice                  | Slice                                                 |
| 25     | Softmax                | Softmax                                               |
| 26     | Upsample               | Resize                                                |
