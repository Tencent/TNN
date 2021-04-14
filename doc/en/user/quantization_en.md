# Model Quantization  

[中文版本](../../cn/user/quantization.md)

## I. Why Quantization
Quantization converts the main operators (Convolution, Pooling, Binary, etc.) in the network from the original floating-point precision to the int8 precision, reducing the model size and improving performance.
PS:
1. For the KL quantization method, you can refer to: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

## II. Compile  
### 1. Build   
```
cd <path_to_tnn>/platforms/linux/
./build_quanttool.sh -c
```
### 2. Output 
    Binary of the quantization tool: <path_to_tnn>/platforms/linux/build/quantization_cmd  

## III. Usage
### 1. Command  
```
./quantization_cmd [-h] [-p] [-m] [-i] [-b] [-w] [-n] [-s] [-r] [-t] <param>
```
### 2. Parameter Description  

|option           |mandatory|with value |description                                      |
|:------------------|:------:|:-----:|:----------------------------------------------|
|-h, --help         |        |       |Output command prompt.                                 |
|-p, --proto        |&radic; |&radic;|Specify tnnproto model description file.                 |
|-m, --model        |&radic; |&radic;|Specify the tnnmodel model parameter file.               |
|-i, --input_path   |&radic; |&radic;|Specify the path of the quantitative input folder. The currently supported formats are: <br>&bull; Text file (the file suffix is ​​.txt) <br>&bull; Common picture format files (file suffix is ​​.jpg .jpeg .png .bmp) <br> All files under this directory will be used as input.|
|-b, --blob_method  |        |&radic;|Specify the feature map quantization method：<br>&bull; 0 Min-Max method (default)<br>&bull; 2 KL method|
|-w, --weight_method|        |&radic;|Specify the quantification method of weights: <br>&bull; 0 Min-Max method (default)<br>&bull; 1 ADMM method|
|-n, --mean         |        |&radic;|Pre-processing, mean operation on each channel of input data, parameter format: 0.0, 0.0, 0.0|
|-s, --scale        |        |&radic;|Pre-processing, scale the input data channels, the parameter format is: 1.0, 1.0, 1.0|
|-r, --reverse_channel|        |&radic;|Pre-processing, valid for picture format files: <br>&bull; 0 use RGB order (default)<br>&bull; 1 use BGR order|
|-t, --merge_type|        |&radic;|Whether use per-tensor or per-channel method when quantifying: <br>&bull; 0 per-channel method (default)<br>&bull; 1 mix method, weights: per-channel, blob: per-tensor.<br>&bull; 2 per-tensor method|  
  
### 3. Quantization Input   
#### 3.1 Select input data    
The input needs to include specific input data, otherwise it will affect the accuracy of the output result, and keep the number of pictures at about 20 ~ 50.
#### 3.2 Input preprocess   
The input data is preprocessed mainly through mean and scale parameters. The formula is:   
input_pre = (input - mean) * scale  

### 4. Quantization Output  
Two files will be generated in the current directory where the command is executed:   
* model_quantized.tnnproto　--　Quantified model description file;
* model_quantized.tnnmodel　--　Quantified model parameter file;

### 5. Note  
（1）-n and -s parameter only works when the input is a picture；  
（2）When the input is a picture，it will be converted to RGB format for processing internally;
 (3) When the input is txt, the input data storage method is NCHW, and of type float. The storage format stores one data in one line, in total of N*C*H*W lines. E.g,
```
0.01
1.1
0.1
255.0
...
```
 (4) scale and mean need to be the value after calculation. For example, 1.0/128.0 is invalid and 0.0078125 is ok.  
