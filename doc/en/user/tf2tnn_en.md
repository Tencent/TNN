# Tensorflow Model to TNN Nodel

[中文版本](../../cn/user/tf2tnn.md)

To convert model file formats such as Pytorch, Tensorflow, and Caffe to TNN, you need to use the corresponding tool to convert from the original format to ONNX model first, which then will be transferred into a TNN model.

With the help of the ONNX open-source community, the open-source converter tf2onnx  can directly convert the Tensorflow model to an ONNX model. The following document will briefly introduce how to use tf2onnx to convert. We strongly recommend that you go to the official website of tf2onnx to understand the detailed usage of tf2onnx, which will help you convert TensorFlow models to TNN models. After using tf2onnx to convert the Tensorflow model to ONNX, it is recommended to refer to the relevant documents of [onnx2tnn](onnx2tnn_en.md) to convert the ONNX model to TNN.


tf2onnx：https://github.com/onnx/tensorflow-onnx


## 1. Environment（Mac and Linux）
- tensorflow (version == 1.15.0)
Recommend to use tensorflow 1.15.0. The compatibility of tensorflow 2.+ is not very stable(bot recommend).
```shell script
pip3 install tensorflow==1.15.0
```

- tf2onnx （version>= 1.5.5）
```shell script
pip3 install tf2onnx
```
- onnxruntime(version>=1.1.0)
```shell script
pip3 install onnxruntime
```

## 2. How to use tf2onnx 

The following is an example command to convert test.pb. It is recommended that you read tf2onnx's README.md file, which contains detailed descriptions of the various parameters of the tool.
``` shell script
python3 -m tf2onnx.convert  --graphdef test.pb   --inputs input_data:0  --outputs pred:0   --opset 11 --output  test.onnx --inputs-as-nchw input_data:0
```

## 3. tf2onnx Operator support

List of operators supported by the tool: [support op](https://github.com/onnx/tensorflow-onnx/blob/master/support_status.md)
