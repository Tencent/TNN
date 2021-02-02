# ONNX(Pytorch) Model to TNN Model

[中文版本](../../cn/user/onnx2tnn.md)

The onnx2tnn is the most important converter in TNN, which converts the ONNX model to a TNN model. The onnx2tnn tool mainly supports CNN common network structure. Because the Pytorch officially supports exporting to ONNX format, we only need to ensure that the ONNX model can be converted into a TNN model so that we could directly guarantee the Pytorch model can be directly converted into a TNN model.

onnx2tnn has an out-of-the-box web version available at https://convertmodel.com/#outputFormat=tnn. Skip the "Environment requirements and Compile" step if you use the web version. The web version converts the model locally, so there is no need to warry about the mode security.

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
onnx=1.6.0  
onnxruntime>=1.1.0   
numpy>=1.17.0  
onnx-simplifier>=0.2.4  
```shell script
pip3 install onnx==1.6.0 onnxruntime numpy onnx-simplifier
```

- cmake （version >= 3.0）
Download the latest cmake from official website, and follow the instructions. It is recommended to use the latest cmake.

### Compile
The onnx2tnn tool runs directly on Mac and Linux with automatic compilation scripts
 ```shell script
cd <path-to-tnn>/tools/onnx2tnn/onnx-converter
./build.sh 
 ```

you could compile the tool manually:
```shell script
# make a new build directory to build
mkdir build
cd build cmake ./../
make -j 4

#copy the so lib
cp ./*.so ./../

#delete build directory
cd ./../
rm -r build
```

##### Manual Compilation

Although we provide the automated script to compile，you could compile the tool manually:

1. cd to the directy
```shell script
cd <path-to-tnn>/tools/onnx2tnn/onnx-converter
```

2. onnx_converter Compile
```shell script
# make a new build directory to build
mkdir build
cd build cmake ./../
make -j 4

#copy the so lib
cp ./*.so ./../

#delete build directory
cd ./../
rm -r build
```

## 2. How to use onnx2tnn 

Check the tool help information
```shell script
python3 onnx2tnn.py -h
```
help information shows as follow:
```text
usage: onnx2tnn.py [-h] [-version VERSION] [-optimize OPTIMIZE] [-half HALF]
                   [-o OUTPUT_DIR]
                   onnx_model_path

positional arguments:
  onnx_model_path     Input ONNX model path

optional arguments:
  -h, --help              show this help message and exit
  -version VERSION        Algorithm version string
  -optimize OPTIMIZE      Optimize model befor convert, 1:default yes, 0:no
  -half HALF              Save model using half, 1:yes, 0:default no
  -o OUTPUT_DIR           the output dir for tnn model
  -align                  align the onnx model with tnn model
  -input_file INPUT_FILE  the input file path which contains the input data for the inference model
  -ref_file   REF_FILE    the reference file path which contains the reference data to compare the results
```


```shell script
python3 onnx2tnn.py model.onnx -version=algo_version -optimize=1 -half=0
```
```text
Parameter:
-version
Version information

-optimize
1（Default）: lossless operator fusion optimisation. For example, BN+Scale..fused into Conv layer；
0 ：If the fusion goes wrong, try set this value. 

-half
1: store in FP16，to reduce the model size.
0（Default）: Stored in fp32.
Note: Whether using fp16 depends on the platform，mobile GPU only allows fp16 calculation

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
The list of operators currently supported by the onnx2tnn tool can be found in [Model Support](support_en.md)
- The tnn2onnx only supports 4-dimensional (nchw) data type.
- It is recommended to set the batch size of the model input to 1, and not recommended to set the batch_size to a relatively large value.
- Asymmetric padding is not supported. (This special situation occurs in "pool5/7x7\_s1\_1" in the inceptionv1 model)
- The upsample layer of onnxruntime version 1.1 and the Pytoch Upsample layer have inconsistent results in the align_corners = 0 mode. Be careful using the onnxruntime calculation result when aligning the results.

# Pytorch model converted to ONNX model

Pytorch supports direct converting from the trained model to the ONNX model, so using the Pytorch's own export method will easily export the Pytorch model to the ONNX model. The following code shows how to export the resnet50 Pytorch model to the ONNX model.
Pytorch also provides more detailed documentation on how to export Pytorch models as ONNX models. For details, please refer to [pytorch export onnx](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

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
Through the code above, after converting the Pytorch model to the ONNX model, you could refer to the onnx2TNN which then converts the onnx model to a TNN model.
