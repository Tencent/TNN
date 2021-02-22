# 模型转换介绍

[English Version](../../en/user/convert_en.md)

<div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/convert.png"/>

目前 TNN 支持业界主流的模型文件格式，包括ONNX、PyTorch、TensorFlow、TesorFlow-Lite 以及 Caffe 等。如上图所示，TNN 将 ONNX 作为中间层，借助于ONNX 开源社区的力量，来支持多种模型文件格式。如果要将PyTorch、TensorFlow 以及 Caffe 等模型文件格式转换为 TNN，首先需要使用对应的模型转换工具，统一将各种模型格式转换成为 ONNX 模型格式，然后将 ONNX 模型转换成 TNN 模型。  

| 原始模型   | 转换工具        | 目标模型 |
|------------|-----------------|----------|
| PyTorch    | pytorch export  | ONNX     |
| TensorFlow | tensorflow-onnx | ONNX     |
| Caffe      | caffe2onnx      | ONNX     |
| ONNX       | onnx2tnn        | TNN      |
| TensorFlow-Lite     | tflite2tnn      | TNN      |
目前 TNN 目前仅支持 CNN 等常用网络结构，RNN、GAN等网络结构正在逐步开发中。

# TNN 模型转换工具

通过上面的模型转换的总体介绍，可以发现如果想将 TensorFlow 模型转换成 TNN 模型需要最少两步，稍显麻烦，所以我们提供了 convert2tnn 工具。这个工具提供了集成的转换工具，可以将 TensorFlow、Caffe 和 ONNX 模型转换成 TNN 模型。由于 PyTorch 可以直接导出为 ONNX 模型，然后再将 ONNX 模型转换成 TNN 模型，所以本工具不再提供对于 PyTorch 模型的模型转换，

大家可以使用 convert2tnn 工具对相关的模型直接进行转换，也可以基于后面文档的相关内容，先将对应的模型转换成 ONNX 模型，然后再将 ONNX 转换成 TNN 模型.

本文中提供了两种方式帮助大家使用 convert2tnn工具：
- 通过 docker image 的方式使用 covnert2tnn 转换工具；
- 手动安装依赖工具和编译工具的方式使用 convert2tnn 转换工具；

## Convert2tnn Docker (推荐)

为了简化 convert2tnn转换工具的安装和编译步骤，目前 TNN 提供了 Dockerfile 文件以及 Docker image 的方式，你可以自己根据 Dockerfile 文件自己构建 docker 镜像，也可以从 Docker Hub 上直接拉取已经构建好的镜像。你可以选择自己喜欢的方式获取 docker 的镜像。

### 拉取构建好的 docker 镜像（推荐）

目前 TNN 已经在 docker hub 上准备好了构建好的 docker image，我们建议直接从 docker hub 上拉取镜像。

```shell script
docker pull turandotkay/tnn-convert
```
同样的，等待一会之后，你可以通过 `docker images` 来查看是否构建成功，如果构建成功之后，会有类似下面的输出信息：
``` text
REPOSITORY                TAG                 IMAGE ID            CREATED             SIZE
turandotkay/tnn-convert   latest              28c93a738b08        15 minutes ago      2.81GB
```
我们发现pull 下来的 docker 镜像的 REPOSIOTY 的名称太长了，我们可以通过下面的命令进行重命名：
```
docker tag turandotkay/tnn-convert:latest tnn-convert:latest
docker rmi turandotkay/tnn-convert:latest
```
此时再次执行 `docker images` 命令，会得到下面的类似的输出：
``` text
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
tnn-convert         latest              28c93a738b08        16 minutes ago      2.81GB
```

#### 更新 docker 镜像
重复 [__拉取构建好的 docker 镜像__](#拉取构建好的-docker-镜像推荐) 中的操作即可

### 构建 docker 镜像(如果上面已经拉取了 image，这一步，可直接跳过)
``` shell script
cd <path-to-tnn>/
docker build -t tnn-convert:latest .
```
docker 会根据 Dockerfile 文件进行构建，这需要等待一会。等构建完成之后，你可以通过下面的命令进行验证是否构建完成。
``` shell script
docker images
```
在输出的列表中会有下面类似的输出，这表明docker 的镜像已经构建好了。
``` text
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
tnn-convert         latest              9fb83110d2c9        26 minutes ago      2.79GB
```



### convert2tnn 工具进行转换

首先验证下 docker 镜像能够正常使用，首先我们通过下面的命令来看下 convert2tnn 的帮助信息：

``` shell script
docker run  -it tnn-convert:latest  python3 ./converter.py -h
```
如果docker 镜像是正确的话，你会得到下面的输出：
```text

usage: convert [-h] {onnx2tnn,caffe2tnn,tf2tnn} ...

convert ONNX/Tensorflow/Caffe model to TNN model

positional arguments:
  {onnx2tnn,caffe2tnn,tf2tnn}
    onnx2tnn            convert onnx model to tnn model
    caffe2tnn           convert caffe model to tnn model
    tf2tnn              convert tensorflow model to tnn model
    tflite2tnn          convert tensorflow-lite model to tnn model

optional arguments:
  -h, --help            show this help message and exit
```
从上面的帮助信息中，我们可以得知，目前 convert2tnn 提供了 3 种模型格式的转换支持。假设我们这里想将 TensorFlow 模型转换成 TNN 模型，我们输入下面的命令继续获得帮助信息：

``` shell script
docker run  -it tnn-convert:latest  python3 ./converter.py tf2tnn -h
```
得到的输出信息如下：
``` text
usage: convert tf2tnn [-h] -tp TF_PATH -in input_info [input_info ...] -on output_name [output_name ...] [-o OUTPUT_DIR] [-v v1.0] [-optimize] [-half] [-align] [-input_file INPUT_FILE_PATH]
                      [-ref_file REFER_FILE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -tp TF_PATH           the path for tensorflow graphdef file
  -in input_info [input_info ...]
                        specify the input name and shape of the model. e.g., -in input1_name:1,128,128,3 input2_name:1,256,256,3
  -on output_name [output_name ...]
                        the tensorflow model's output name. e.g. -on output_name1 output_name2
  -o OUTPUT_DIR         the output tnn directory
  -v v1.0               the version for model
  -optimize             optimize the model
  -half                 optimize the model
  -align                align the onnx model with tnn model
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference data to compare the results.
```
通过上面的输出，可以发现针对 TF 模型的转换，convert2tnn 工具提供了很多参数，我们一次对下面的参数进行解释：

- tp 参数（必须）
    通过 “-tp” 参数指定需要转换的模型的路径。目前只支持单个 TF模型的转换，不支持多个 TF 模型的一起转换。
- in 参数（必须）
    通过 “-in” 参数指定模型输入，例如：-in input_name_1:1,128,128,3 input_name_2:1,256,256,3。
- on 参数（必须）
    通过 “-on” 参数指定模型输出的名称，例如: -on output_name1 output_name2
- output_dir 参数：
    可以通过 “-o <path>” 参数指定输出路径，但是在 docker 中我们一般不使用这个参数，默认会将生成的 TNN 模型放在当前和 TF 模型相同的路径下。
- optimize 参数（可选）
    可以通过 “-optimize” 参数来对模型进行优化，**我们强烈建议你开启这个选项，只有在开启这个选项模型转换失败时，我们才建议你去掉 “-optimize” 参数进行重新尝试**。
- v 参数（可选）
    可以通过 -v 来指定模型的版本号，以便于后期对模型进行追踪和区分。
- half 参数（可选）
    可以通过 -half 参数指定，模型数据通过 FP16 进行存储，减少模型的大小，默认是通过 FP32 的方式进行存储模型数据的。
- align 参数（可选）
    可以通过 -align 参数指定，将 转换得到的 TNN 模型和原模型进行对齐，确定 TNN 模型是否转换成功。__align 只支持 FP32 模型的校验，所以使用 align 的时候不能使用 half__
- input_file 参数（可选）
    可以通过 -input_file 参数指定模型对齐所需要的输入文件的名称，输入需要遵循如下[格式](#输入)。生成输入的代码可以[参考](#生成输入或输出文件示例代码)。
- ref_file 参数（可选）
    可以通过 -ref_file 参数指定待对齐的输出文件的名称，输出需遵循如下[格式](#输出)。生成输出的代码可以[参考](#生成输入或输出文件示例代码)。


**当前 convert2tnn 的模型只支持 graphdef 模型，不支持 checkpoint 以及 saved_model 格式的文件，如果想将 checkpoint 或者 saved_model 的模型进行转换，可以参看下面[tf2tnn](./tf2tnn.md)的部分，自行进行转换。**

下面我们通过一个例子来展示如何将 TF 模型转换到 TNN 模型，

``` shell script
docker run --volume=$(pwd):/workspace -it tnn-convert:latest  python3 ./converter.py tf2tnn \
    -tp /workspace/test.pb \
    -in "input0:1,32,32,3 input2:1,32,32,3" \
    -on output0 output1 \
    -v v2.0 \
    -optimize \
    -align \
    -input_file /workspace/in.txt \
    -ref_file /workspace/ref.txt
```

由于 convert2tnn工具是部署在 docker 镜像中的，如果要进行模型的转换,需要先将模型传输到 docker 容器中。我们可以通过 docker run 的参数--volume 将包含模型的模型挂载到 docker 容器的某个路径下。上面的例子中是将执行shell 的当前目录（pwd）挂载到 docker 容器中的 "/workspace” 文件夹下面。当然了测试用到的test.pb 也**必须执行 shell 命令的当前路径下**。执行完成上面的命令后，convert2tnn 工具会将生成的 TNN 模型存放在 test.pb文件的同一级目录下，当然了生成的文件也就是在当前目录下。

上面的文档中只是介绍了 TensorFlow 的模型的转换，其他模型的使用也是类似的，可以自行通过转换工具的帮助信息的提醒进行使用，我这里不在对这些转换命令进行详细的说明，只是简单的将这些转换命令列出来，你可以仿照着进行转换。

``` shell script
# convert onnx
docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py onnx2tnn \
    /workspace/mobilenetv3-small-c7eb32fe.onnx \
    -optimize \
    -v v3.0 \
    -align  \
    -input_file /workspace/in.txt \
    -ref_file /workspace/ref.txt

# convert caffe
docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py caffe2tnn \
    /workspace/squeezenet.prototxt \
    /workspace/squeezenet.caffemodel \
    -optimize \
    -v v1.0 \
    -align  \
    -input_file /workspace/in.txt \
    -ref_file /workspace/ref.txt
    
# convert tflite
docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py tflite2tnn \
    /workspace/mobilenet_v1_1.0_224.tflite \
    -v v1.0 \
    -align  \
    -input_file /workspace/in.txt \
    -ref_file /workspace/ref.txt


```

## Convert2tnn 手动安装
如果你不想使用 docker 镜像的方式，也可以在自己的开发机上安装 convert2tnn 的依赖工具，并根据相关的说明进行编译，也可以同样使用 convert2tnn 工具机型模型转换。

convert2tnn 的完整环境搭建包含下面的所有的工具的安装和编译。如果你只想转换某一类的模型，你只需要安装转换对应模型转换的依赖工具。例如你只想转换 caffe 的模型，你就不需要安装 转换 TensorFlow 模型依赖的工具。同理你需要转换 TensorFlow 的模型，就可以不用安装 Caffe 模型转换的依赖工具。但是 ONNX 模型依赖工具和安装和编译都是必须的。

针对 Linux 系统下的环境配置，我使用 Centos 7.2 为例，Ubuntu 系统也可以适用，只要将相应的安装命令修改为 Ubuntu 上的对应命令即可。  

### 环境搭建及编译
#### 1. ONNX模型转换工具搭建（必须）
- 安装protobuf(version >= 3.4.0)  
Macos:
```shell script
brew install protobuf
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
- 安装 python 依赖库
onnx=1.6.0  
onnxruntime>=1.1.0   
numpy>=1.17.0  
onnx-simplifier>=0.2.4  
protobuf>=3.4.0
```shell script
pip3 install onnx==1.6.0 onnxruntime numpy onnx-simplifier protobuf
```

- cmake （version >= 3.0）
从的官网下载最新版本的 cmake，然后按照文档安装即可。建议使用最新版本的 cmake。

##### 编译
onnx2tnn 工具在 Mac 以及 Linux 上有自动编译脚本直接运行就可以。
 ```shell script
cd <path-to-tnn>/tools/convert2tnn
./build.sh
 ```

#### 2. TensorFlow 模型转换（可选）


- tensorflow (version == 1.15.0)
建议使用 TensorFlow 1.15.0 的版本，目前 TensorFlow 2.+ 的版本的兼容性不好， 不建议使用。
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

#### 3. Caffe 模型转换（可选）

- 安装protobuf(version >= 3.4.0)  

Macos:
```shell script
brew install protobuf
```

Linux:

对于 linux 系统，我们建议参考 protobuf 的官方[README](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md)文档，直接从源码进行安装。  

如果你使用的是 Ubuntu 系统可以使用下面的指令进行安装：
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

#### convert2tnn 工具的使用
配置后上面的环境依赖之后，就可以使用 convert2tnn 进行相应模型的转换

```shell script
cd <path_to_tnn_root>/tools/convert2tnn/
python3 converter.py -h
```
执行上面的命令会打印下面的信息。目前 convert2tnn 提供了三个子命令，分别对相应的模型进行转换。

```text
usage: convert [-h] {onnx2tnn,caffe2tnn,tf2tnn} ...

convert ONNX/Tensorflow/Caffe model to TNN model

positional arguments:
  {onnx2tnn,caffe2tnn,tf2tnn}
    onnx2tnn            convert onnx model to tnn model
    caffe2tnn           convert caffe model to tnn model
    tf2tnn              convert tensorflow model to tnn model

optional arguments:
  -h, --help            show this help message and exit
```
- ONNX模型转换
如果想相对 ONNX 模型进行转换，可以直接使用 onnx2tnn 的子命令来查看帮助信息。

```shell script
python3 converter.py onnx2tnn -h
```
usage 信息如下：
```text
usage: convert onnx2tnn [-h] [-in input_info [input_info ...]] [-optimize]
                        [-half] [-v v1.0.0] [-o OUTPUT_DIR] [-align]
                        [-input_file INPUT_FILE_PATH]
                        [-ref_file REFER_FILE_PATH] [-debug]
                        onnx_path

positional arguments:
  onnx_path             the path for onnx file

optional arguments:
  -h, --help            show this help message and exit
  -in input_info [input_info ...]
                        specify the input name and shape of the model. e.g.,
                        -in input1_name:1,3,128,128 input2_name:1,3,256,256
  -optimize             optimize the model
  -half                 save model using half
  -v v1.0.0             the version for model
  -o OUTPUT_DIR         the output tnn directory
  -align                align the onnx model with tnn model
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for
                        the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference
                        data to compare the results.
  -debug                Turn on the switch to debug the model.
```
示例：
```shell script
python3 converter.py onnx2tnn \
    ~/mobilenetv3/mobilenetv3-small-c7eb32fe.onnx.onnx \
    -optimize \
    -v=v3.0 \
    -o ~/mobilenetv3/ \
    -align \
    -input_file in.txt \
    -ref_file ref.txt
```

- caffe2tnn

Caffe 格式转换

目前 convert2tnn 的工具目前只支持最新版本的 Caffe 的文件格式,所以如果想将 Caffe 模型转换为 TNN 模型。需要先将老版本的 Caffe 网络和模型转换为新版. Caffe 自带了工具可以把老版本的

Caffe 网络和模型转换为新版本的格式. 具体的使用方式如下:
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


```shell script
python3 converter.py caffe2tnn -h
```
usage 信息如下：
```text
usage: convert caffe2tnn [-h] [-o OUTPUT_DIR] [-v v1.0] [-optimize] [-half]
                         prototxt_file_path caffemodel_file_path

positional arguments:
  prototxt_file_path    the path for prototxt file
  caffemodel_file_path  the path for caffemodel file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR         the output tnn directory
  -v v1.0               the version for model, default v1.0
  -optimize             optimize the model
  -half                 save model using half
  -align                align the onnx model with tnn model
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for
                        the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference
                        data to compare the results.
```
示例：
```shell script
python3 converter.py caffe2tnn \
    ~/squeezenet/squeezenet.prototxt \
    ~/squeezenet/squeezenet.caffemodel \
    -optimize \
    -v v1.0 \
    -o ~/squeezenet/ \
    -align \
    -input_file in.txt \
    -ref_file ref.txt
```
- tensorflow2tnn

当前 convert2tnn 的模型只支持 graphdef 模型，不支持 checkpoint 以及 saved_model 格式的文件，如果想将 checkpoint 或者 saved_model 的模型进行转换，可以参看下面的 tf2onnx 的部分，自行进行转换。

``` shell script
python3 converter.py tf2tnn -h
```
usage 信息如下：
```text
usage: convert tf2tnn [-h] -tp TF_PATH -in input_info [input_info ...] -on output_name [output_name ...] [-o OUTPUT_DIR] [-v v1.0] [-optimize] [-half] [-align] [-input_file INPUT_FILE_PATH]
                      [-ref_file REFER_FILE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -tp TF_PATH           the path for tensorflow graphdef file
  -in input_info [input_info ...]
                        specify the input name and shape of the model. e.g., -in input1_name:1,128,128,3 input2_name:1,256,256,3
  -on output_name [output_name ...]
                        the tensorflow model's output name. e.g. -on output_name1 output_name2
  -o OUTPUT_DIR         the output tnn directory
  -v v1.0               the version for model
  -optimize             optimize the model
  -half                 optimize the model
  -align                align the onnx model with tnn model
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference data to compare the results.
```
- tensorflow-lite2tnn

当前 tensorflow-lite2tnn 的转换支持tflite格式文件，从而方便移动端部署。

``` shell script
python3 converter.py tflite2tnn -h
```
usage 信息如下：
```
usage: convert tflite2tnn [-h] TF_PATH [-o OUTPUT_DIR] [-v v1.0] [-align]

optional arguments:
  -h, --help            show this help message and exit
   TF_PATH           the path for tensorflow-lite graphdef file
  -o OUTPUT_DIR         the output tnn directory
  -v v1.0               the version for model
  -align                align the onnx model with tnn model
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for
                        the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference
                        data to compare the results.
```
示例：
```shell script
python3 converter.py tflite2tnn \
    ~/tf-model/test.tflite \
    -o ~/tf-model/ \
    -align \
    -input_file in.txt \
    -ref_file ref.txt
```

## 输入输出文件格式示例
### 输入
```text

输入数量 
输入名称 shape维度个数 具体shape信息 输入数据类型
输入数据 
输入名称 shape维度个数 具体shape信息 输入数据类型
输入数据 
......

例如
 2 
 in0 4 1 3 1 1 3
 2 
 4 
 3 
 in1 4 1 2 2 1 0
 0.1 
 0.2 
 0.3 
 0.4 


提示：
如果输入数据是 float, 输入数据类型可以用 0 表示
如果输入数据是 int  , 输入数据类型可以用 3 表示

```

### 输出
```text

输出数量 
输出名称 shape维度个数 具体shape信息 输出数据类型
输出数据 
输出名称 shape维度个数 具体shape信息 输出数据类型
输出数据
......

例如
 2 
 out0 2 1 3 0
 0.1 
 0.2 
 0.3 
 out1 4 1 2 2 1 0
 0.1 
 0.2 
 0.3 
 0.4 


提示：
如果输出数据是 float, 输出数据类型可以用 0 表示
如果输出数据是 int  , 输出数据类型可以用 3 表示

```

### 生成输入或输出文件示例代码
```python
"""

输入或输出至少包含以下三个部分：
name -> 类型：str。用于标识该输出的名称。
shape -> 类型：list。用于描述输出的维数。
tensor -> 类型：numpy.ndarray。存放输出数据。

提示：
对于输出文件，如果 shape 维度不足四维的，需要使用 1 进行填充。例如：(n, c) => (n, c, 1, 1)。
输入则没有这项要求。

假设有两个输出，它们分别如下：
输出一组成部分：
name_1
shape_1
tensor_1

输出二组成部分：
name_2
shape_2
tensor_2

你可以参考以下代码将输出写到文件中

"""

# 输出数量
num_output = 2

# 输出文件保存路径
output_path = "output.txt"

with open(output_path, "w") as f:
    # 保存输出数量
    f.write("{}\n" .format(num_output))

    # 保存第一个输出
    description_1 = "{} {} " .format(name_1, len(shape_1))
    for dim in shape_1:
        description_1 += "{} " .format(dim)
    data_type_1 = 0 if tensor.dtype == np.float else 3
    description_1 += "{}" .format(data_type)
    f.write(description_1 + "\n")
    np.savetxt(f, tensor_1.reshape(-1), fmt="%0.18f")

    # 保存第二个输出
    description_2 = "{} {} " .format(name_2, len(shape_2))
    for dim in shape_2:
        description_2 += "{} " .format(dim)
    data_type_2 = 0 if tensor.dtype == np.float else 3
    description_2 += "{}" .format(data_type)
    f.write(description_2 + "\n")
    np.savetxt(f, tensor_2.reshape(-1), fmt="%0.18f")


```


## 模型转换详细介绍
convert2tnn 只是对多种模型转换的工具的封装，根据第一部分 “模型转换介绍”中原理说明，你也可以先将原始模型转换成 ONNX，然后再将 ONNX 模型转换成 TNN 模型。我们提供了如何手动的将 Caffe、PyTorch、TensorFlow 模型转换成 ONNX 模型，然后再将 ONNX 模型转换成 TNN 模型的文档。如果你在使用 convert2tnn 转换工具遇到问题时，我们建议你了解下相关的内容，这有可能帮助你更加顺利的进行模型转换。

- [onnx2tnn](onnx2tnn.md)
- [pytorch2tnn](onnx2tnn.md)
- [tf2tnn](tf2tnn.md)
- [caffe2tnn](caffe2tnn.md)
- [tflite2tnn](tflite2tnn.md)

