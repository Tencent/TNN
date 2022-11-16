# ONNX(Pytorch) 模型转换为 TNN 模型

[English Version](../../en/user/onnx2tnn_en.md)

onnx2tnn 是 TNN 中最重要的模型转换工具，它的主要作用是将 ONNX 模型转换成 TNN 模型格式。目前 onnx2tnn 工具支持主要支持 CNN 常用网络结构。由于 Pytorch 模型官方支持支持导出为 ONNX 模型，并且保证导出的 ONNX 模型和原始的 Pytorch 模型是等效的，所以我们只需要保证 ONNX 模型能够转换为 TNN 模型，就直接能够保证 Pytorch 直接转换为 TNN 模型。

onnx2tnn 有开箱即用的 [网页版](https://convertmodel.com/#outputFormat=tnn)，使用网页版的用户可以跳过环境搭建和编译的步骤。网页版在浏览器本地完成模型转换，不会将模型上传到云端，用户可以不用担心模型安全。

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
onnx=1.6.0  
onnxruntime>=1.1.0   
numpy>=1.17.0  
onnx-simplifier>=0.2.4  
protobuf>=3.4.0
requests
```shell script
pip3 install onnx==1.6.0 onnxruntime numpy onnx-simplifier protobuf requests
```

- cmake （version >= 3.0）
从的官网下载最新版本的 cmake，然后按照文档安装即可。建议使用最新版本的 cmake。

### 编译
onnx2tnn 工具在 Mac 以及 Linux 上有自动编译脚本直接运行既可以。
 ```shell script
cd <path-to-tnn>/tools/onnx2tnn/onnx-converter
./build.sh 
 ```

如果想自己进行编译的话，可以按照下面的步骤自己进行编译。
```shell script
# 新建build目录进行编译
mkdir build
cd build cmake ./../
make -j 4

#复制生成的so库
cp ./*.so ./../

#删除build目录
cd ./../
rm -r build
```

##### 手动编译

虽然我们提供了自动化编译的脚本，当然了你也可以自己手动的进行编译。手动编译过程如下所示：

1. 切换到工具目录
```shell script
cd <path-to-tnn>/tools/onnx2tnn/onnx-converter
```

2. onnx_converter 的编译
```shell script
# 新建build目录进行编译
mkdir build
cd build cmake ./../
make -j 4

#复制生成的so库
cp ./*.so ./../

#删除build目录
cd ./../
rm -r build
```

## 2. onnx2tnn 工具的使用

首先查看工具的帮助信息：
```shell script
python3 onnx2tnn.py -h
```
help 信息如下所示：
```text
usage: onnx2tnn.py [-h] [-version VERSION] [-optimize OPTIMIZE] [-half HALF]
                   [-o OUTPUT_DIR] [-input_shape INPUT_SHAPE]
                   onnx_model_path

positional arguments:
  onnx_model_path       Input ONNX model path

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
python3 onnx2tnn.py model.onnx -version=v1.0 -optimize=1 -half=0 -o out_dir/ -input_shape input:1,3,224,224
```
```text
参数说明：
-version
模型版本号，便于后续算法进行跟踪

-optimize
1（默认，开启）: 用于对模型进行无损融合优化，如BN+Scale等f融合进Conv层；
0 ：如果融合报错可以尝试设为此值

-half
1: 转为FP16模型存储，减小模型大小。
0（默认，不开启）: 按照FP32模型存储。
Note: 实际计算是否用FP16看各个平台特性决定，移动端GPU目前仅支持FP16运算

-o
output_dir : 指定 TNN 模型的存放的文件夹路径，该文件夹必须存在

-input_shape
模型输入的 shape，用于模型动态 batch 的情况
```


## 3. 算子支持情况以及使用限制
目前 onnx2tnn 的工具能够支持的算子列表，可以在[模型支持](support.md)
- 目前 tnn2onnx 只支持 4 维(nchw)的数据类型。
- 建议将模型输入的 batch size 设置为 1，不建议将 batch_size设置为比较大的值。
- 目前暂不支持 pool 中的 pad 为非对称的情况。（经测试inceptionv1 模型中的 "pool5/7x7\_s1\_1"会出现这种特殊的情况）
- onnxruntime1.1版本 的 Upsample层与Pytoch的Upsample层在 align_corners = 0 模式下结果不一致，对齐结果时谨慎使用onnxruntime的计算结果。

# Pytorch 模型转换为 ONNX 模型

Pytorch 支持直接将训练好的模型转换为 ONNX 模型是，所以使用 Pytorch 自带的导出方法，会很方便的将Pytorch 模型导出为 ONNX 模型，下面的代码展示如何将 resnet50的 Pytorch 模型导出为 ONNX 模型。
Pytorch 也提供了更详细的文档来介绍如何将 Pytorch 模型导出为 ONNX 模型，具体可以参考[pytorch export onnx](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

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
通过上面的代码，将 Pytorch 模型转换为 ONNX 模型之后，可以参考上面“ONN转换为 TNN 模型”的章节，将生成的 ONNX 模型转换为 TNN 模型。
