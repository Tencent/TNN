# Model Verification Tool

[中文版本](../../cn/development/model_check.md)

## I. Function
Check whether the output of the model on corresponding platform (OpenCL, Metal, Cuda, ARM, HuaweiNPU) is correct.

## II. Compile
To compile the model_check tool, the following macro must be set to ON:  
* Turn on the following options to compile TNN (For the compilation method, please refer to [Compile TNN](../user/compile_en.md))
* `TNN_CPU_ENABLE`  
* `TNN_MODEL_CHECK_ENABLE`
* set for corresponding device，such as `TNN_OPENCL_ENABLE`, `TNN_ARM_ENABLE`

## III. Usage
### 1. Command
```
./model_check [-h] [-p] <tnnproto> [-m] <tnnmodel> [-d] <device> [-i] <input> [-f] <refernece> [-e] [-n] <val> [-s] <val> [-o] [-b]
```
### 2. Parameter Description
|option           |mandatory|with value |description                                       |  
|:------------------|:------:|:-----:|:-------------------------------------------|  
|-h, --help         |        |       |Output command prompt.                                |  
|-p, --proto        |&radic; |&radic;|Specify tnnproto model description file.                   |   
|-m, --model        |&radic; |&radic;|Specify the tnnmodel model parameter file.                   |  
|-d, --device       |&radic; |&radic;|Specify the platform on which the model is executed, such as OPENCL, ARM, METAL, CUDA, HUAWEI_NPU etc.    |  
|-i, --input_path   |        |&radic;|Specify the input file. The currently supported formats are:<br>&bull; Text file (the file suffix is ​​.txt). The format is the same as the input file dumped by model converter tool. <br>&bull; Common picture format files (file suffix is ​​.jpg .jpeg .png .bmp)<br>If not specified, (-1, 1) will be used for random input|  
|-f, --ref          |        |&radic;|Use the specified output to compare the results. The currently supported formats are:<br>&bull; Text file (file suffix is ​​.txt), the format is the same as the output file dumped by model converter tool.|
|-e, --end          |        |       |Only check output of model.                           |  
|-n, --mean         |        |&radic;|Pre-processing, mean operation on each channel of input data, parameter format: 0.0, 0.0, 0.0|  
|-s, --scale        |        |&radic;|Pre-processing, scale the input data channels, the parameter format is: 1.0, 1.0, 1.0|  
|-o, --output       |        |       |Whether to save the final output.                           |  
|-b, --batch        |        |       |Check the result of each batch. (Not finished yet) |  



## IV. Execute the Script
### 1. Android
#### 1.1 Prepare models
Copy the tnnproto and tnnmodel files of the model to be verified into `<path_to_tnn>/platforms/android/models` and rename them to` test.tnnproto` and `test.tnnmodel`
#### 1.2 Execute the script
`` `
cd <path_to_tnn>/platforms/android/
./model_check_android.sh -c -m <tnnproto> -p
`` `
### 2. Linux
#### 2.1. Compile the script
`` `
cd <path_to_tnn>/platforms/linux/
./build_model_check.sh -c
`` `
#### 2.2. Execute the command
`` `
<path_to_tnn>/platforms/linux/build/model_check -p <path_to_tnnproto> -m <path_to_tnnmodel> -d <DEVICE>
`` `

## V. Tool Restrictions
* Currently the tool only supports fp32 model verification;
* For per-layer model checking, only the fp32 precision is supported. For only-output model checking, the precision is decided by device automatically.
