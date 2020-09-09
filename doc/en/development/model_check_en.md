# Model Verification Tool

[中文版本](../../cn/development/model_check.md)

## I. Function
Check whether the output of the model on corresponding platform (OpenCL, Metal, Cuda, ARM) is correct.

## II. Compile
To compile the model_check tool, the following macro must be set to ON:  
* Turn on the following options to compile TNN (For the compilation method, please refer to [Compile TNN](../user/compile_en.md))
* `TNN_NAIVE_ENABLE`  
* `TNN_MODEL_CHECK_ENABLE`
* set for corresponding device，such as `TNN_OPENCL_ENABLE`, `TNN_ARM_ENABLE`

## III. Usage
### 1. Command
```
./model_check [-h] [-p] [-m] [-d] [-i] [-n] [-s] [-o] [-f] <param>
```
### 2. Parameter Description
|option           |mandatory|with value |description                                       |  
|:------------------|:------:|:-----:|:-------------------------------------------|  
|-h, --help         |        |       |Output command prompt.                                |  
|-p, --proto        |&radic; |&radic;|Specify tnnproto model description file.                   |   
|-m, --model        |&radic; |&radic;|Specify the tnnmodel model parameter file.                   |  
|-d, --device       |&radic; |&radic;|Specify the platform on which the model is executed, such as OPENCL, ARM, METAL, etc.    |  
|-i, --input_path   |        |&radic;|Specify the input file. The currently supported formats are:<br>&bull; Text file (the file suffix is ​​.txt)<br>&bull; Common picture format files (file suffix is ​​.jpg .jpeg .png .bmp)<br>If not specified, (-1, 1) will be used for random input|  
|-n, --mean         |        |&radic;|Pre-processing, mean operation on each channel of input data, parameter format: 0.0, 0.0, 0.0|  
|-s, --scale        |        |&radic;|Pre-processing, scale the input data channels, the parameter format is: 1.0, 1.0, 1.0|  
|-o, --output       |        |       |Whether to save the final output.                           |  
|-f, --ref          |        |&radic;|Use the specified output to compare the results. The currently supported formats are:<br>&bull; Text file (file suffix is ​​.txt), data storage is in NCHW format, separated by newline.|



## IV. Execute the Script
### 1. Android
#### 1.1 Prepare models
Copy the tnnproto and tnnmodel files of the model to be verified into `<path_to_tnn>/platforms/android/modles` and rename them to` test.tnnproto` and `test.tnnmodel`
#### 1.2 Execute the script
`` `
cd <path_to_tnn>/platforms/android/
./model_check_android.sh -c -p
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
* At present, only the fp32 results can be verified;
