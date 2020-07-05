# Tensorflow 模型转换为 TNN 模型

[English Version](../../en/user/tf2tnn_en.md)

要将 Tensorflow 模型转换为 TNN 模型，首先将 Tensorflow 模型转换为 ONNX 模型，然后再将ONNX 模型转换为 TNN 模型。

将 Tensorflow 模型转换为ONNX，我们借助于 ONNX 开源社区的力量，ONNX 开源社区提供的开源的转换工具 tf2onnx，可以直接将 Tensorflow 模型转换为 ONNX 模型。在下面的文档中，会简单的介绍如何使用 tf2onnx 进行转换。我们强烈建议你去 tf2onnx 的官网，去了解 tf2onnx 的详细用法，这会帮助你更好的将 TensorFlow模型转换为 TNN 模型。当使用 tf2onnx 将 Tensorflow 模型转换为 ONNX 之后，建议参考 [onnx2tnn](onnx2tnn.md) 的相关文档，将 ONNX 模型转换为 TNN。


tf2onnx项目地址：https://github.com/onnx/tensorflow-onnx


## 1. 环境搭建（Mac and Linux）
- tensorflow (version == 1.15.0)
建议使用 tensorflow 1.15.0 的版本，目前 tensorflow 2.+ 的版本的兼容性不好， 不建议使用。
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

## 2. tf2onnx 工具的使用

下面是对 test.pb 进行转换的命令，使用起来非常方便。建议大家阅读 tf2onnx 的 README.md 文件，里面有详细的对该工具各个参数的说明。
``` shell script
python3 -m tf2onnx.convert  --graphdef test.pb   --inputs input_data:0[1,28,28,3]  --outputs pred:0   --opset 11 --output  test.onnx --inputs-as-nchw input_data:0
```

## 3. tf2onnx 支持的算子

该工具支持的算子的列表: [support op](https://github.com/onnx/tensorflow-onnx/blob/master/support_status.md)
