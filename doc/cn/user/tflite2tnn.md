#Tensorflow-lite 模型转换为 TNN 模型

[English Version](../../en/user/onnx2tnn_en.md)

tflite2tnn 是 TNN 中最重要的模型转换工具，它的主要作用是将 TensorFlow-Lite 模型转换成 TNN 模型格式。

TensorFlow-Lite 模型可以直接转换成TNN模型。接下来文档将简要介绍如何用 tflite2tnn 工具转换模型。
## 1. 环境搭建及编译
### 环境搭建
以下的环境搭建适用于 Macos 以及 Linux 系统，其中 Linux 以 centos7.2 为例。

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
Centos:
```shell script
yum install  python3 python3-devel
```

- 安装 python 依赖库  
numpy>=1.17.0  
protobuf>=3.4.0
```shell script
pip3 install numpy  protobuf
```

- cmake （version >= 3.0）
从的官网下载最新版本的 cmake，然后按照文档安装即可。建议使用最新版本的 cmake。

### 编译
tflite2tnn 工具在 Mac 以及 Linux 上有自动编译脚本直接运行既可以。
 ```shell script
cd <path-to-tnn>/tools/convert2tnn
./build.sh 
 ```

## 2. tflite2tnn 工具的使用

首先查看工具的帮助信息：
```shell script
python3 converter.py tflite2tnn  -h
```
help 信息如下所示：
```text
usage: converter.py tflite2tnn [-h] tflitemodel_path [-version VERSION] [-o OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -version VERSION      Algorithm version string
  -optimize OPTIMIZE    If the model has fixed input shape, use this option to optimize the model for speed. On the other hand, if the model has dynamic input shape, dont use this option. It may cause warong result
  -half HALF            Save model using half, 1:yes, 0:default no
  -o OUTPUT_DIR         the output dir for tnn model
  -input_shape INPUT_SHAPE
                        manually-set static input shape, useful when the input
                        shape is dynamic
```


```shell script
python3 converter.py tflite2tnn  test.tflite
```
```text
参数说明：
-version
模型版本号，便于后续算法进行跟踪
-o
output_dir : 指定 TNN 模型的存放的文件夹路径，该文件夹必须存在
```


## 3. 算子支持情况以及使用限制
目前 tflite2tnn 的工具能够支持的算子列表，可以在[支持列表](support_tflite_mode.md)