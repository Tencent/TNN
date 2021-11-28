# Caffe 模型转换为 ONNX 模型

[English Version](../../en/user/caffe2tnn_en.md)

要将 Caffe 模型转换为 TNN 模型，首先将 Caffe 模型转换为 ONNX 模型，然后再将ONNX 模型转换为 TNN 模型。

将 Caffe 模型转换为ONNX，我们借助于 caffe2onnx 工具, 它可以直接将 Caffe 模型转换为 ONNX 模型。在下面的文档中，会简单的介绍如何使用 caffe2onnx进行转换，然后建议参考 [onnx2tnn](onnx2tnn.md) 的相关文档，再将 ONNX 模型转换为 TNN。


## 1. 环境搭建(Mac and Linux)

- 安装protobuf(version >= 3.4.0)  

Macos:
```shell script
brew install protobuf
```

Linux:

对于 linux 系统，我们建议参考protobuf 的官方[README](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md)文档，直接从源码进行安装。  

如果你使用的是Ubuntu 系统可以使用下面的指令进行安装：
```shell script
sudo apt-get install libprotobuf-dev protobuf-compiler
```

- 安装python (version >=3.6)  

Macos
```shell script
brew install python3
```
centos:
```shell script
yum install  python3 python3-devel
```

- onnx(version == 1.6.0)
```shell script
pip3 install onnx==1.6.0
```

- numpy(version >= 1.17.0)
```shell script
pip3 install numpy
```

## 2. caffe2onnx 工具使用
- 进入工具目录
``` shell script
cd <tnn_root_path>/tools/caffe2onnx/
```
- caffe 格式转换

目前 caffe2onnx 的工具目前只支持最新版本的 caffe 的格式,所以在使用 caffe2onnx
工具之前需要将老版本的 caffe 网络和模型转换为新版. caffe 自带了工具可以把老版本的
caffe 网络和模型转换为新版本的格式. 具体的使用方式如下:
```shell script
upgrade_net_proto_text [老prototxt] [新prototxt]
upgrade_net_proto_binary [老caffemodel] [新caffemodel]
```
修改后的输入的格式如下所示:

```text
layer {
  name: "data"
  type: "input"
  top: "data"
  input_param { shape: { dim: 1 dim: 3 dim: 224 dim: 224 } }
}
```
- caffe2onnx 工具的使用

```shell script
python3 convert2onnx.py ./test.prototxt ./test.caffemodel -o ./test.onnx -align -input_file=in.txt -ref_file=ref.txt
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
  -h, --help            show this help message and exit
  -o OUTPUT_DIR         the output tnn directory
  -v v1.0               the version for model, default v1.0
  -optimize             If the model has fixed input shape, use this option to optimize the model for speed. On the other hand, if the model has dynamic input shape, dont use this option. It may cause wrong result
  -half                 save model using half
  -align                align the onnx model with tnn model
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for
                        the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference
                        data to compare the results.
```
注意：当前仅支持单输入单输出模型和单输入多输出模型。 align 只支持 FP32 模型的校验，所以使用 align 的时候不能使用 half。

## 3. caffe2onnx 支持的算子

| Number | caffe layer          | onnx operator                                       |
| ------ | -------------------- | --------------------------------------------------- |
| 1      | BatchNorm            | BatchNormalization                                  |
| 2      | BatchNorm + Scale    | BatchNormalization                                  |
| 3      | Concat               | Concat                                              |
| 4      | Convolution          | Conv                                                |
| 5      | ConvolutionDepthwise | Conv                                                |
| 6      | Crop                 | Slice                                               |
| 7      | Deconvolution        | ConvTranspose                                       |
| 8      | DetectionOutput      | DetectionOutput(customer defination)                |
| 9      | Dropout              | Dropout                                             |
| 10     | Eltwise              | Mul/Add/Max                                         |
| 11     | Flatten              | Reshape                                             |
| 12     | InnerProduct         | Reshape + Gemm                                      |
| 13     | LRN                  | LRN                                                 |
| 14     | MaxUnPool            | MaxUnPool                                           |
| 15     | MVN                  | InstanceNorm                                        |
| 16     | PReLU                | PRelu                                               |
| 17     | Permute              | Transpose                                           |
| 18     | Pooling              | MaxPool/AveragePool/GlobalMaxPool/GlobalAveragePool |
| 19     | Power                | Mul/Add/Pow                                         |
| 20     | PriorBox             | PriorBox(customer defination)                       |
| 21     | ReLU                 | Relu/LeakyRelu                                      |
| 22     | ReLU6                | Clip                                                |
| 23     | Reshape              | Reshape                                             |
| 24     | Scale                | Mul + Reshape                                       |
| 25     | ShuffleChannel       | Reshape + Transpose + Reshape                       |
| 26     | Sigmoid              | Sigmoid                                             |
| 27     | Slice                | Slice                                               |
| 28     | Softmax              | Softmax                                             |
| 29     | Upsample             | Resize                                              |

