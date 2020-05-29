# Model quantization tool  
## I. Why quantization
Quantization converts the main operators (Convolution, Pooling, Binary, etc.) in the network from the original floating-point precision to the int8 precision, reducing the model size and improving performance.
PS:
1. For the KL quantization method, you can refer to: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

## II. Compile  
### 1. build with script  
```
cd <path_to_tnn>/platforms/linux/
./build_quanttool.sh -c
```
### 2. output of the build
    Binary of the quantization tool: <path_to_tnn>/platforms/linux/build/quantization_cmd  

## III. Usage
### 1. Command  
```
./quantization_cmd [-h] [-p] [-m] [-i] [-b] [-w] [-n] [-s] [-c] <param>
```
### 2. parameter description  

|option           |mandatory|with value |description                                      |
|:------------------|:------:|:-----:|:----------------------------------------------|
|-h, --help         |        |       |Output command prompt.                                 |
|-p, --proto        |&radic; |&radic;|Specify rapidproto model description file.                 |
|-m, --model        |&radic; |&radic;|Specify the rapidmodel model parameter file.               |
|-i, --input_path   |&radic; |&radic;|Specify the path of the quantitative input folder. The currently supported formats are: <br>&bull; Text file (the file suffix is ​​.txt) <br>&bull; Common picture format files (file suffix is ​​.jpg .jpeg .png .bmp) <br> All files under this directory will be used as input.|
|-b, --blob_method  |        |&radic;|Specify the feature map quantization method：<br>&bull; 0 Min-Max method (default)<br>&bull; 2 KL method|
|-w, --weight_method|        |&radic;|Specify the quantification method of weights: <br>&bull; 0 Min-Max method (default)<br>&bull; 1 ADMM method|
|-n, --mean         |        |&radic;|
Pre-processing, mean operation on each channel of input data, parameter format: 0.0, 0.0, 0.0|
|-s, --scale        |        |&radic;|Pre-processing, scale the input data channels, the parameter format is: 1.0, 1.0, 1.0|
|-c, --merge_channel|        |&radic;|Whether to calculate all the channels together when quantifying the feature map, otherwise it is calculated separately for each channel.|  
  
### 3. Quantization input   
#### 3.1 Select input data    
The input needs to include specific input data, otherwise it will affect the accuracy of the output result, and keep the number of pictures at about 20 ~ 50.
#### 3.2 Input preprocess   
The input data is preprocessed mainly through mean and scale parameters. The formula is:   
input_pre = (input - mean) * scale  
### 4. Output  
Two files will be generated in the current directory where the command is executed:   
* model_quantized.rapidproto　--　Quantified model description file;
* model_quantized.rapidmodel　--　Quantified model parameter file;
