# Inner Model Converter Tool

[中文版本](../../cn/user/model_converter.md)

## I. Function
+ Convert models among three version of TNN models:
    + RapidnetV1：old model version (RPN)
    + TNN：the open source model version 
    + RapidnetV3：Encrypted version of TNN
+ Dump model information. ie, model version, data type to store weights;

## II. Compile
To compile the model_check tool, the following macro must be set to ON. (For the compilation method, please refer to [Compile TNN](../user/compile_en.md))
* `TNN_MODEL_CONVERT_ENABLE`  

## III. Usage
### 1. Command
```
./model_convert [-h] [-i] [-p] <proto_path> [-m] <model_path> [-v] <version>
```
### 2. Parameter Description
|option           |mandatory|with value |description                              |  
|:------------------|:------:|:-----:|:-------------------------------------------|  
|-h, --help         |        |       |Output command prompt.                      |  
|-i, --info         |        |       |Dump model information.                     |   
|-p, --proto        |&radic; |&radic;|Specify tnnproto model description file.                   |   
|-m, --model        |&radic; |&radic;|Specify the tnnmodel model parameter file.                   |  
|-v, --version      |        |&radic;|Specify the model version to save.<br>&bull; 0：RapidnetV1 <br>&bull; 1：TNN <br>&bull; 2：RapidnetV3 (default)| 

## IV. Execute the Script
### 1. Linux
#### 1.1. Compile the script
```
cd <path_to_tnn>/platforms/linux/
./build_model_convert.sh -c
```
#### 1.2. Execute the command
+ Model Convert
```
<path_to_tnn>/platforms/linux/build/model_converter -p <path_to_tnnproto> -m <path_to_tnnmodel> -v <version>
```
+ Model Information Dump
```
<path_to_tnn>/platforms/linux/build/model_converter -p <path_to_tnnproto> -m <path_to_tnnmodel> -i
```

## V. Tool Restrictions
* The proto and model must match;
