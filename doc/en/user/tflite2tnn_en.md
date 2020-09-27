# Tensorflow-lite Model to TNN Nodel

[中文版本](../../cn/user/tflite2tnn.md)

To convert Tensorflow-lite model file  to TNN, you need to use the corresponding tool to convert from the original format to TNN model.

The  Tensorflow-lite model can directly convert  to an TNN model. The following document will briefly introduce how to use tflite2tnn to convert.

## 1. Environment requirements and Compile
### Environment requirements
The following environment is suitable for Macos and Linux systems.
The example is based on centos7.2.

-Install protobuf (version >= 3.4.0)

Macos:
```shell script
brew install protobuf
```

Linux:

For linux systems, we recommend to refer to the official [README](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) document of protobuf and install directly from the source code.

If you are using Ubuntu system, you can use the following instructions to install:

```shell script
sudo apt-get install libprotobuf-dev protobuf-compiler
```



- Install python (version >= 3.6)

Macos
```shell script
brew install python3
```
Centos:
```shell script
yum install  python3 python3-devel
```

- Install python dependencies
numpy>=1.17.0   
protobuf>=3.4.0
```shell script
pip3 install numpy protobuf
```

- cmake （version >= 3.0）
Download the latest cmake from official website, and follow the instructions. It is recommended to use the latest cmake.

### Compile
The tflite2tnn tool runs directly on Mac and Linux with automatic compilation scripts
 ```shell script
cd <path-to-tnn>/tools/convert2tnn
./build.sh 
 ```

## 2. How to use tflite2tnn 

Check the tool help information
```shell script
python3 converter.py tflite2tnn  -h
```
help information shows as follow:
```text
usage: converter.py tflite2tnn [-h] tflitemodel_path [-version VERSION] [-o OUTPUT_DIR]

optional arguments:
  -h, --help              show this help message and exit
  -version VERSION        Algorithm version string
  -o OUTPUT_DIR           the output dir for tnn model
  -align                  align the onnx model with tnn model
  -input_file INPUT_FILE  the input file path which contains the input data for the inference model
  -ref_file   REF_FILE    the reference file path which contains the reference data to compare the results
```


```shell script
python3 converter.py tflite2tnn  test.tflite
```
```text
Parameter:
-version
Version information

-o
output_dir : The directory the model to be saved in，the directory must exit already.

-align
model align, if you want to use it, you can add '-align' in your command

-input_file
input_file : The path of input file, which will be used in model align

-ref_file
reference_file : The path of reference file, which will be used in model align. Compare tnn's output and reference file.
```


## 3. Operator support and usage restrictions
List of operators supported by the tool: [tflite support list](support_tflite_mode_en.md)
