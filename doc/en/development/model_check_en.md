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
./model_check [-h] [-p] <tnnproto> [-m] <tnnmodel> [-d] <device> [-i] <input> [-f] <refernece> [-e] [-n] <val> [-s] <val> [-o] [-b] -sp [precision]
```
### 2. Parameter Description
|option   |mandatory|with value |description                                       |  
|:-------:|:-------:|:-----:|:-------------------------------------------|  
|-h       |         |       |Output command prompt.                                |  
|-p       |&radic;  |&radic;|Specify tnnproto model description file.                   |   
|-m       |&radic;  |&radic;|Specify the tnnmodel model parameter file.                   |  
|-d       |&radic;  |&radic;|Specify the platform on which the model is executed, such as OPENCL, ARM, METAL, CUDA, HUAWEI_NPU etc.    |  
|-i       |         |&radic;|Specify the input file. The currently supported formats are:<br>&bull; Text file (the file suffix is ​​.txt). The format is the same as the input file dumped by model converter tool. <br>&bull; Common picture format files (file suffix is ​​.jpg .jpeg .png .bmp)<br>If not specified, (-1, 1) will be used for random input|  
|-f       |         |&radic;|Use the specified output to compare the results. The currently supported formats are:<br>&bull; Text file (file suffix is ​​.txt), the format is the same as the output file dumped by model converter tool.|
|-e       |         |       |Only check output of model.                           |  
|-n       |         |&radic;|Pre-processing, mean operation on each channel of input data, parameter format: 0.0, 0.0, 0.0|  
|-s       |         |&radic;|Pre-processing, scale the input data channels, the parameter format is: 1.0, 1.0, 1.0|  
|-o       |         |       |Whether to save the final output.                           |  
|-b       |         |       |Check the result of each batch.  |  
|-sp      |         |&radic;|Set the precision of device(AUTO/NORMAL/HIGH/LOW)|  

Note: the formula of bias and scale is: y=(x-bias)*scale

### 3. Txt file format
```
<blob_num_s>
<blob1_name> <dim_size_n1> <dim0> <dim1> ... <dim(n1-1)> <data type>
<data>
...
<data>
<blob2_name> <dim_size_n2> <dim0> <dim1> ... <dim(n2-1)> <data type>
<data>
...
<data>
...
<blob(s)_name> <dim_size_ns> <dim0> <dim1> ... <dim(ns-1)> <data type>
<data>
...
<data>
```

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
* For per-layer model checking, only the fp32 precision is supported. For only-output model checking, the precision is decided by device automatically.
