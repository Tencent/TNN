# How to Create a TNN Model

[中文版本](../../cn/user/convert.md)

## Overview

<div align=left><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/user/resource/convert.png"/>

TNN currently supports the industry's mainstream model file formats, including ONNX, Pytorch, Tensorflow and Caffe. As shown in the figure above, TNN utilizes ONNX as the intermediate port to support multiple model file formats. 
To convert model file formats such as Pytorch, Tensorflow, TensorFlow-Lite, and Caffe to TNN, you need to use corresponding tool to convert from the original format to ONNX model first, which then will be transferred into a TNN model.

| Source Model   | Convertor        | Target Model |
|------------|-----------------|----------|
| Pytorch    | pytorch export directly | ONNX     |
| Tensorflow | tensorflow-onnx | ONNX     |
| Caffe      | caffe2onnx      | ONNX     |
| ONNX       | onnx2tnn        | TNN      |
| TensorFlow-Lite     | tflite2tnn      | TNN      |

At present, TNN only supports common network structures such as CNN. Networks like RNN and GAN are under development.

## TNN Model Converter

Through the general introduction above, you can find that it takes at least two steps to convert a Tensorflow model into a TNN model. So we provide an integrated tool, convert2tnn, to simplify. The converter can convert Tensorflow, Caffe and ONNX models into TNN models by just one single operation. Since Pytorch can directly export ONNX models, this tool no longer provides support for the Pytorch model.

You can use the convert2tnn tool to directly convert the models to TNN, or you can first convert the corresponding model into ONNX format and then convert it to a TNN model based on the documents.

This article provides two ways to help you use the convert2tnn tool:
- Use covnert2tnn via Docker image;
- Manually install dependencies and tools to use convert2tnn converter;

### Convert2tnn Docker (Recommend)

In order to simplify the installation and compilation steps of the convert2tnn converter, TNN provides a Dockerfile and a Docker image method. You can build the Docker image yourself based on the Dockerfile file, or you can directly pull the built Docker image from Docker Hub. You can choose the way you like to obtain the Docker image.

#### Pull from the Docker Hub (Recommend)

At present, TNN has prepared a built Docker image on Docker Hub. We suggest pulling the Docker image directly from Docker Hub. 

```shell script
docker pull ccr.ccs.tencentyun.com/qcloud/tnn-convert
```
 After waiting for a while, you can check through `docker images` command. If successful, there will be output similar to the following:

```text
REPOSITORY                                  TAG                 IMAGE ID            CREATED             SIZE
ccr.ccs.tencentyun.com/qcloud/tnn-convert   latest              66605e128277        2 hours ago         3.54GB
```

If the REPOSITORY name is too long, rename it with the following command:
```text
docker tag ccr.ccs.tencentyun.com/qcloud/tnn-convert tnn-convert:latest
docker rmi ccr.ccs.tencentyun.com/qcloud/tnn-convert
```
After renaming the docker image, you can check through the `docker images` command. If successful, there will be output similar to the following:
```text
REPOSITORY                           TAG                 IMAGE ID            CREATED             SIZE
tnn-convert                          latest              66605e128277        2 hours ago         3.54GB
```

#### Build Docker image (If the image is pulled through previous step, skip this part)

``` shell script
cd <path-to-tnn>/
docker build -t tnn-convert:latest.
```

Docker will build a Docker image based on the Dockerfile, which needs a while to complete. After the construction is completed, you can verify whether the installation process is successful by the following command.

``` shell script
docker images
```

There should be similar output as shown below, which indicates that the Docker image has been built.
``` text
REPOSITORY TAG IMAGE ID CREATED SIZE
tnn-convert latest 9e2a73fbfb3b 18 hours ago 2.53GB
```

#### Use convert2tnn to convert

First, verify that the Docker image's status. Let's take a look at the help information of convert2tnn by entering the following command:

``` shell script
docker run -it tnn-convert:latest python3 ./converter.py -h
```

If the Docker image is correct, you will obtain the following output:

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
From above，we can know that currently convert2tnn provides conversion support for 3 model formats. Suppose we want to convert the Tensorflow model to a TNN model here, we enter the following subcommand to continue to get help information:

``` shell script
docker run  -it tnn-convert:latest  python3 ./converter.py tf2tnn -h
```
The output shows below：
``` text
usage: convert tf2tnn [-h] -tp TF_PATH -in input_info [input_info ...] -on
                      output_name [output_name ...] [-o OUTPUT_DIR] [-v v1.0]
                      [-optimize] [-half] [-align [{None,output,all}]]
                      [-input_file INPUT_FILE_PATH]
                      [-ref_file REFER_FILE_PATH] [-debug] [-int8]

optional arguments:
  -h, --help            show this help message and exit
  -tp TF_PATH           the path for tensorflow graphdef file
  -in input_info [input_info ...]
                        specify the input name and shape of the model. e.g.,
                        -in input1_name:1,128,128,3 input2_name:1,256,256,3
  -on output_name [output_name ...]
                        the tensorflow model's output name. e.g. -on
                        output_name1 output_name2
  -o OUTPUT_DIR         the output tnn directory
  -v v1.0               the version for model
  -optimize             If the model has fixed input shape, use this option to
                        optimize the model for speed. On the other hand, if
                        the model has dynamic input shape, dont use this
                        option. It may cause warong result
  -half                 save the model using half
  -align [{None,output,all}]
                        align the onnx model with tnn model. e.g., if you want
                        to align the last output, you can use '-align' or
                        '-align output'; if the model is not align, you can
                        use '-align all' to address the first unaligned layer
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for
                        the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference
                        data to compare the results.
  -debug                Turn on the switch to debug the model.
  -int8                 save model using dynamic range quantization. use int8
                        save, fp32 interpreting
```
Here are the explanations for each parameter:

- tp parameter (required)
    Use the "-tp" parameter to specify the path of the model to be converted. Currently only supports the conversion of a single TF model, does not support the conversion of multiple TF models together.
- in parameter (required)
    Specify the name of the model input through the "-in" parameter, for example "-in input1_name:1,128,128,3 input2_name:1,256,256,3"
- on parameter (required)
    Specify the name of the model output through the "-on" parameter, for example "-on output_name1 output_name2"
- output_dir parameter:
    You can specify the output path through the "-o <path>" parameter, but we generally do not apply this parameter in docker. By default, the generated TNN model will be placed in the same path as the TF model.
- optimize parameter (optional)
    You can optimize the model with the "-optimize" parameter. **If the model has fixed input shape, use this option to optimize the model for speed. On the other hand, if the model has dynamic input shape, dont use this option. It may cause warong result**.
- v parameter (optional)
    You can use -v to specify the version number of the model to facilitate later tracking and differentiation of the model.
- half parameter (optional)
   You can save the model with the "-half" parameter. The model data will be stored in FP16 to reduce the size of the model by setting this parameter. By default, the model data is stored in FP32.
- align parameter (optional)
    You can align the model with the "-align" parameter. Compare TNN model and original model to determine whether TNN model is correct. If you remove "-align", model align will not run; if you use "-align" or "-align output", this tool will compare the last output of TNN model and original model; if the model is not align, you can use '-align all' to address the first unaligned layer.
- fold_const parameter (optional)
    You can optimize the model with the "-fold_const" parameter. Enable tf constant_folding transformation before conversion.
- input_file parameter (optional)
    Specify the input file's name which will be used by model_check through the "-input_file" parameter. This is [input format](#Input).
- ref_file parameter (optional)
    Specify the reference file's name which will be used by model_check through the "-ref_file" parameter. This is [output format](#Output). 
- debug (optional)
  controls whether to output the full log of model conversions.
- int8 (optional)
  You can quantize the model to 8 bits using "-int8". The model will be saved using 8 bits, model are interpreted to FP32 when loaded. Only Conv, LSTM, MatMul are quantized.


**Current convert2tnn input model only supports graphdef format，does not support checkpoint or saved_model format. Refer to [tf2tnn](./tf2tnn_en.md) to transfer checkpoint or saved_model models.**

Here is an example of converting a TF model in a TNN model

``` shell script
docker run --volume=$(pwd):/workspace -it tnn-convert:latest  python3 ./converter.py tf2tnn -tp=/workspace/test.pb -in=input0,input2 -on=output0 -v=v2.0 -optimize 
```

Since the convert2tnn tool is deployed in the Docker image, if you want to convert the model, you need to first push the model into the Docker container. We can use the docker run parameter --volume to mount certain path in the Docker container. In the above example, the current directory (pwd) for executing the shell is under the "/workspace" folder in the Docker container. The test.pb used in the test therefore **must be executed under the current path of the shell command**. After executing the above command, the convert2tnn tool will store the generated TNN model in the same level directory of the test.pb file. 

The above information only introduces the conversion for Tensorflow's models. It is similar for other model formats. You can use the conversion tool's note to remind yourself. These subcommands are listed below:

``` shell script
# convert onnx
docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py onnx2tnn /workspace/mobilenetv3-small-c7eb32fe.onnx -optimize -v=v3.0
# convert caffe
docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py caffe2tnn /workspace/squeezenet.prototxt /workspace/squeezenet.caffemodel -optimize -v=v1.0

# convert tflite
docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py tflite2tnn \
    /workspace/mobilenet_v1_1.0_224.tflite \
    -v v1.0 \
    -align  \
    -input_file /workspace/in.txt \
    -ref_file /workspace/ref.txt


```

### Manual Convert2tnn Installation
You can also install the dependencies of convert2tnn on your development machine manually and compile it according to the relevant instructions. 

If you only want to convert the models of certain types, you just need to install the corresponding dependent tools. For example, if you only want to convert the caffe model, you do not need to install the tools that the Tensorflow model depends on. Similarly, if you need to convert Tensorflow's model, you don't need to install Caffe model conversion dependent tools. However, the ONNX model depends on tools and installation and compilation are required.

The example runs on Centos 7.2. It can also be applied to Ubuntu, as long as the corresponding installation command is modified to the corresponding command on Ubuntu.

#### Build the environment
##### 1. Build ONNX model conversion tool (Required)

- Install protobuf(version >= 3.4.0) 
 
Macos:
```shell script
brew install protobuf
```
## Set proxy (optional)
export http_proxy=http://{addr}:{port}
export https_proxy=http://{addr}:{port}
## Compile

- install python (version >=3.6)  

Macos
```shell script
brew install python3
```
centos:
```shell script
yum install  python3 python3-devel
```

- install python dependencies
onnx=1.6.0  
onnxruntime>=1.1.0   
numpy>=1.17.0  
onnx-simplifier>=0.2.4 
requests
```shell script
pip3 install onnx==1.6.0 onnxruntime numpy onnx-simplifier requests
```

- cmake （version >= 3.0）
Download the latest cmake，and follow the instructions in official documents to install.
It is recommended to use the latest version.

###### Compile
The onnx2tnn tool can run directly on Mac and Linux with automatic compilation scripts.
 ```shell script
cd <path-to-tnn>/tools/onnx2tnn/onnx-converter
./build.sh
 ```

##### 2. Convert the TensorFlow Model (Optional)

-tensorflow (version == 1.15.0)
It is recommended to use tensorflow version 1.15.0. The current version of tensorflow 2.+ is not compatible and is not recommended.
```shell script
pip3 install tensorflow==1.15.0
```

-tf2onnx (version>= 1.5.5)
```shell script
pip3 install tf2onnx
```
-onnxruntime(version>=1.1.0)
```shell script
pip3 install onnxruntime
```
##### 3. Convert the Caffe model (Optional)

-Install protobuf (version >= 3.4.0)

Macos:
```shell script
brew install protobuf
```

Linux:

For linux system, we suggest referring to the official [README](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) document of protobuf and install directly from the source code.

If you are using Ubuntu system, you can use the following instructions to install:

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

#### convert2tnn Tool Usage
After meeting the requirements, convert2tnn could convert the models.

```shell script
cd <path_to_tnn_root>/tools/convert2tnn/
python3 converter.py -h
```
Execute the command above will output information below. There 3 options at present.

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
- ONNX model conversion
If you want to convert ONNX models, you can directly choose the onnx2tnn option to view the help information.

```shell script
python3 converter.py onnx2tnn -h
```
usage information：
```text
usage: convert onnx2tnn [-h] [-in input_info [input_info ...]] [-optimize]
                        [-half] [-v v1.0.0] [-o OUTPUT_DIR]
                        [-align [{None,output,all}]] [-align_batch]
                        [-input_file INPUT_FILE_PATH]
                        [-ref_file REFER_FILE_PATH] [-debug] [-int8]
                        onnx_path

positional arguments:
  onnx_path             the path for onnx file

optional arguments:
  -h, --help            show this help message and exit
  -in input_info [input_info ...]
                        specify the input name and shape of the model. e.g.,
                        -in input1_name:1,3,128,128 input2_name:1,3,256,256
  -optimize             If the model has fixed input shape, use this option to
                        optimize the model for speed. On the other hand, if
                        the model has dynamic input shape, dont use this
                        option. It may cause warong result
  -half                 save model using half
  -v v1.0.0             the version for model
  -o OUTPUT_DIR         the output tnn directory
  -align [{None,output,all}]
                        align the onnx model with tnn model. e.g., if you want
                        to align the last output, you can use '-align' or
                        '-align output'; if the model is not align, you can
                        use '-align all' to address the first unaligned layer
  -align_batch          align the onnx model with tnn model and check mutli
                        batch
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for
                        the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference
                        data to compare the results.
  -debug                Turn on the switch to debug the model.
  -int8                 save model using dynamic range quantization. use int8
                        save, fp32 interpreting
```
Example:
```shell script
python3 converter.py onnx2tnn ~/mobilenetv3/mobilenetv3-small-c7eb32fe.onnx.opt.onnx -optimize -v=v3.0 -o ~/mobilenetv3/ 
```

- caffe2tnn

caffe format conversion

The convert2tnn currently only supports the latest version of Caffe file format, so if you want to convert Caffe model to TNN model. You need to convert the old version caffe network and model into new version first. Caffe comes with such tools.

The caffe network and model are converted to the new version format. The specific usage is as follows:

```shell script
upgrade_net_proto_text [old prototxt] [new prototxt]
upgrade_net_proto_binary [old caffemodel] [new caffemodel]
```

The format after modification:

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
usage information：
```text
usage: convert caffe2tnn [-h] [-o OUTPUT_DIR] [-v v1.0] [-optimize] [-half]
                         [-align [{None,output,all}]]
                         [-input_file INPUT_FILE_PATH]
                         [-ref_file REFER_FILE_PATH] [-debug] [-int8]
                         prototxt_file_path caffemodel_file_path

positional arguments:
  prototxt_file_path    the path for prototxt file
  caffemodel_file_path  the path for caffemodel file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR         the output tnn directory
  -v v1.0               the version for model, default v1.0
  -optimize             If the model has fixed input shape, use this option to
                        optimize the model for speed. On the other hand, if
                        the model has dynamic input shape, dont use this
                        option. It may cause warong result
  -half                 save model using half
  -align [{None,output,all}]
                        align the onnx model with tnn model. e.g., if you want
                        to align the last output, you can use '-align' or
                        '-align output'; if the model is not align, you can
                        use '-align all' to address the first unaligned layer
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for
                        the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference
                        data to compare the results.
  -debug                Turn on the switch to debug the model.
  -int8                 save model using dynamic range quantization. use int8
                        save, fp32 interpreting
```
Example：
```shell script
python3 converter.py caffe2tnn ~/squeezenet/squeezenet.prototxt ~/squeezenet/squeezenet.caffemodel -optimize -v=v1.0 -o ~/squeezenet/ 
```
- tensorflow2tnn

The current convert2tnn model only supports the graphdef model, but does not support checkpoint or saved_model format files. If you want to convert the checkpoint or saved_model model, you can refer to the tf2onnx section below to convert it yourself.

``` shell script
python3 converter.py tf2tnn -h
```
usage information：
```text
usage: convert tf2tnn [-h] -tp TF_PATH -in input_info [input_info ...] -on
                      output_name [output_name ...] [-o OUTPUT_DIR] [-v v1.0]
                      [-optimize] [-half] [-align [{None,output,all}]]
                      [-input_file INPUT_FILE_PATH]
                      [-ref_file REFER_FILE_PATH] [-debug] [-int8]

optional arguments:
  -h, --help            show this help message and exit
  -tp TF_PATH           the path for tensorflow graphdef file
  -in input_info [input_info ...]
                        specify the input name and shape of the model. e.g.,
                        -in input1_name:1,128,128,3 input2_name:1,256,256,3
  -on output_name [output_name ...]
                        the tensorflow model's output name. e.g. -on
                        output_name1 output_name2
  -o OUTPUT_DIR         the output tnn directory
  -v v1.0               the version for model
  -optimize             If the model has fixed input shape, use this option to
                        optimize the model for speed. On the other hand, if
                        the model has dynamic input shape, dont use this
                        option. It may cause warong result
  -half                 save the model using half
  -align [{None,output,all}]
                        align the onnx model with tnn model. e.g., if you want
                        to align the last output, you can use '-align' or
                        '-align output'; if the model is not align, you can
                        use '-align all' to address the first unaligned layer
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for
                        the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference
                        data to compare the results.
  -debug                Turn on the switch to debug the model.
  -int8                 save model using dynamic range quantization. use int8
                        save, fp32 interpreting
```
Example：
```shell script
python3 converter.py tf2tnn -tp ~/tf-model/test.pb -in=input0,input2 -on=output0 -v=v2.0 -optimize -o ~/tf-model/
```
- tensorflow-lite2tnn

The current tensorflow2tnn only supports the tflite format model which is  to facilitate mobile deployment.

``` shell script
python3 converter.py tflite2tnn -h
```
usage information：
```
usage: convert tflite2tnn [-h] [-o OUTPUT_DIR] [-v v1.0] [-half]
                          [-align ALIGN] [-input_file INPUT_FILE_PATH]
                          [-ref_file REFER_FILE_PATH] [-debug] [-int8]
                          tf_path

positional arguments:
  tf_path               the path for tensorflow-lite graphdef file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR         the output tnn directory
  -v v1.0               the version for model
  -half                 optimize the model
  -align ALIGN          align the onnx model with tnn model. e.g., if you want
                        to align the final output, you can use -align outputif
                        you want to align whole model, you can use -align all
  -input_file INPUT_FILE_PATH
                        the input file path which contains the input data for
                        the inference model.
  -ref_file REFER_FILE_PATH
                        the reference file path which contains the reference
                        data to compare the results.
  -debug                Turn on the switch to debug the model.
  -int8                 save model using dynamic range quantization. use int8
                        save, fp32 interpreting
```
Example：
```shell script
python3 converter.py tflite2tnn  ~/tf-model/test.tflite  -o ~/tf-model/
```

## Input and Output File Example
### Input
```text

The number of input 
input_name input_shape_size input_info input_data_type
input_data 
input_name input_shape_size input_info input_data_type
input_data
......

Example
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


Tips：
If input data type is float, you can use 0 to specify input_data_type.
If input data type is int,   you can use 3 to specify input_data_type.

```

### Output
```text


The number of output 
output_name output_shape_size output_info output_data_type
output_data 
output_name output_shape_size output_info output_data_type
output_data
......

Example
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


Tips：
If output data type is float, you can use 0 to specify output_data_type.
If output data type is int,   you can use 3 to specify output_data_type.

```

### The Code Used to Generate Input or Output File
```python
def write_pytorch_data(output_path, data, data_name_list):
    """
    Save the data of Pytorch needed to align TNN model.

    The input and output names of pytorch model and onnx model may not match,
    you can use Netron to visualize the onnx model to determine the data_name_list.

    The following example converts ResNet50 to onnx model and saves input and output:
    >>> from torchvision.models.resnet import resnet50
    >>> model = resnet50(pretrained=False).eval()
    >>> input_data = torch.randn(1, 3, 224, 224)
    >>> input_names, output_names = ["input"], ["output"]
    >>> torch.onnx.export(model, input_data, "ResNet50.onnx", input_names=input_names, output_names=output_names)
    >>> with torch.no_grad():
    ...     output_data = model(input_data)
    ...
    >>> write_pytorch_data("input.txt", input_data, input_names)
    >>> write_pytorch_data("output.txt", output_data, output_names)

    :param output_path: Path to save data.
    :param data: The input or output data of Pytorch model.
    :param data_name_list: The name of input or output data. You can get it after visualization through Netron.
    :return:
    """

    if type(data) is not list and type(data) is not tuple:
        data = [data, ]
    assert len(data) == len(data_name_list), "The number of data and data_name_list are not equal!"
    with open(output_path, "w") as f:
        f.write("{}\n" .format(len(data)))
        for name, data in zip(data_name_list, data):
            data = data.numpy()
            shape = data.shape
            description = "{} {} ".format(name, len(shape))
            for dim in shape:
                description += "{} ".format(dim)
            data_type = 0 if data.dtype == np.float32 else 3
            fmt = "%0.6f" if data_type == 0 else "%i"
            description += "{}".format(data_type)
            f.write(description + "\n")
            np.savetxt(f, data.reshape(-1), fmt=fmt)


def write_tensorflow_data(output_path, data, data_name_list, data_usage=1):
    """
    Save the data of TensoFlow needed to align TNN model.

    :param output_path: Path to save data. "You should use input.txt or output.txt to name input or output data"
    :param data: The input or output data of TensorFlow model.
    :param data_name_list: The name of input or output data. You can get it after visualization through Netron.
    :param data_usage: Specify the data usage. If the data is input data, data_usage=0;
                       if the data if outptu data, data_usage=1.
    :return:
    """
    def convert_nhwc(data):
        assert len(data.shape) <= 4
        if len(data.shape) == 2:
            return data
        orders = (0, 2, 1) if len(data.shape) == 3 else (0, 2, 3, 1)
        return data.transpose(orders)

    if type(data) is not list and type(data) is not tuple:
        data = [data, ]
    assert len(data) == len(data_name_list), "The number of data and data_name_list are not equal!"
    with open(output_path, "w") as f:
        f.write("{}\n" .format(len(data)))
        for name, data in zip(data_name_list, data):
            data = convert_nhwc(data) if data_usage == 0 else data
            shape = data.shape
            description = "{} {} ".format(name, len(shape))
            for dim in shape:
                description += "{} ".format(dim)
            data_type = 0 if data.dtype == np.float32 else 3
            fmt = "%0.6f" if data_type == 0 else "%i"
            description += "{}".format(data_type)
            f.write(description + "\n")
            np.savetxt(f, data.reshape(-1), fmt=fmt)



```

## Model Conversion Details
convert2tnn is just an encapsulation of a variety of tools for model converting. According to the principles explained in the previous part "Introduction to model conversion", you can also convert the original model into ONNX first, and then convert the ONNX model into a TNN model. We provide documentation on how to manually convert Caffe, Pytorch, TensorFlow models into ONNX models, and then convert ONNX models into TNN models. If you encounter problems when using the convert2tnn converter, we recommend that you understand the relevant content, which may help you to use the tool more smoothly.

- [onnx2tnn](onnx2tnn_en.md)
- [pytorch2tnn](onnx2tnn_en.md)
- [tf2tnn](tf2tnn_en.md)
- [caffe2tnn](caffe2tnn_en.md)
- [tflite2tnn](tflite2tnn_en.md)
