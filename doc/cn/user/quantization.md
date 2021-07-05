# 模型量化  

[English Version](../../en/user/quantization_en.md)

## 一、量化的作用  
量化将网络中主要算子（Convolution，Pooling，Binary等）由原先的浮点计算转成低精度的Int8计算，减少模型大小并提升性能。  
PS：    
1、关于KL量化方法，可以参考：http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf  

## 二、编译  
### 1. 编译脚本  
```
cd <path_to_tnn>/platforms/linux/
./build_quanttool.sh -c
```
### 2. 编译输出  
量化模型命令：<path_to_tnn>/platforms/linux/build/quantization_cmd  
## 三、量化工具的使用  
### 1. 命令  
```
./quantization_cmd [-h] [-p] [-m] [-i] [-b] [-w] [-n] [-s] [-r] [-t] <param>
```
### 2. 参数说明  

|命令参数           |是否必须|带参数 |参数说明                                       |
|:------------------|:------:|:-----:|:----------------------------------------------|
|-h, --help         |        |       |输出命令提示。                                 |
|-p, --proto        |✅ |✅|指定tnnproto模型描述文件。                   |
|-m, --model        |✅ |✅|指定tnnmodel模型参数文件。                   |
|-i, --input_path   |✅ |✅|指定量化输入文件夹路径。目前支持格式为：<br>&bull; 文本文件（文件后缀为.txt）<br>&bull; 常用图片格式文件（文件后缀为 .jpg .jpeg .png .bmp）<br>会将此目录下面的所有文件作为输入。|
|-b, --blob_method  |        |✅|指定feature map的量化方法：<br>&bull; 0 Min-Max方法（默认）<br>&bull; 2 KL方法|
|-w, --weight_method|        |✅|指定weights的量化方法：<br>&bull; 0 Min-Max方法（默认）<br>&bull; 1 ADMM方法|
|-n, --bias         |        |✅|预处理，仅对输入为图片时起作用。对输入数据各通道进行bias操作，参数格式为：0.0,0.0,0.0|
|-s, --scale        |        |✅|预处理，仅对输入为图片时起作用。对输入数据各通道进行scale操作，参数格式为：1.0,1.0,1.0|
|-r, --reverse_channel|        |✅|预处理，仅对输入为图片时起作用：<br>&bull; 0 使用RGB顺序（默认）<br>&bull; 1 使用BGR顺序|
|-t, --merge_type|        |✅|在量化的时候采用Per-Tensor还是Per-Channel的方式。<br>&bull; 0 Per-Channel方法（默认）<br>&bull; 1 混合方法，weights采用Per-Channel，blob采用Per-Tensor。<br>&bull; 2 Per-Tensor方法|  
  
### 3. 量化输入   
#### 3.1 输入数据的选取   
输入数据需要包括典型的输入，否则影响输出结果的精度，图片数量在20~50左右。  
#### 3.2 输入预处理    
对图片的输入数据进行预处理，主要通过bias和scale参数进行。公式为：   
input_pre = (input - bias) * scale  

### 4. 命令输出  
在执行命令的当前目录下会生成两个文件：    
* model_quantized.tnnproto　--　量化后的模型描述文件；
* model_quantized.tnnmodel　--　量化后的模型参数文件；

### 5. 注意事项  
（1）-n和-s参数仅对输入为图片的时候有作用；  
（2）输入为图片时，内部会转为RGB格式进行处理；  
（3）输入为txt时，输入数据存储方式是NCHW，且为float类型。存储格式为，一行存储一个数据，总共N\*C\*H\*W行。例如，  
```
0.01
1.1
0.1
255.0
...
```
（4）scale和mean的值必须是计算之后的值，不能使用公式，例如1.0/128.0就是无效的，而0.0078125就是可以的。
