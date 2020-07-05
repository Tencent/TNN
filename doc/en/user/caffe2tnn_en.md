# Caffe Model to ONNX Model

[中文版本](../../cn/user/caffe2tnn.md)

To convert the Caffe model to a TNN model, first convert the Caffe model to an ONNX model, which is then converted to a TNN model.

 We use the caffe2onnx tool to convert the Caffe model to ONNX. In the following document, it will briefly introduce how to use caffe2onnx to convert, and then it is recommended to refer to [onnx2tnn](onnx2tnn_en.md)to convert the ONNX model to TNN.


## 1. Environment(Mac and Linux)

- install protobuf(version >= 3.4.0)  

Macos:
```shell script
brew install protobuf
```

Linux:

For Linux system，we recommend following protobuf's official [README](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md).  

If you are using the Ubuntu system, use the instructions below to install:
```shell script
sudo apt-get install libprotobuf-dev protobuf-compiler
```

- Install python (version >=3.6)  

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

## 2. caffe2onnx tool usage
- cd in tool directory
``` shell script
cd <tnn_root_path>/tools/caffe2onnx/
```
- Caffe format conversion

At present, the caffe2onnx tool currently only supports the latest version of caffe format. So before using caffe2onnx, you need to upgrade the old version of the caffe network and model to the latest version. Caffe comes with the tool to convert from old version caffe model and network to a new version.
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
- caffe2onnx tool usage

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
  -h, --help            show this help message and exit
  -o ONNX_FILE          the path for generate onnx file
  -align                align the onnx model with tnn model
  -input_file in.txt    the input file path which contains the input data for the inference model
  -ref_file   ref.txt   the reference file path which contains the reference data to compare the results
```

## 3. caffe2onnx operator support

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