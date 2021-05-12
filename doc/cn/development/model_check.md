# 模型结果校验

[English Version](../../en/development/model_check_en.md)

## 一、工具的作用
校验对应平台（OpenCL，Metal，Cuda，ARM，HuaweiNPU）的模型输出结果是否正确。

## 二、编译
编译model_check工具需要将以下宏设置为ON：  
* 打开以下选项编译TNN（编译方法参照[TNN编译文档](../user/compile.md))
* `TNN_CPU_ENABLE`  
* `TNN_MODEL_CHECK_ENABLE`
* 对应device的宏，如`TNN_OPENCL_ENABLE`, `TNN_ARM_ENABLE`

## 三、校验工具使用
### 1. 命令
```
./model_check [-h] [-p] <tnnproto> [-m] <tnnmodel> [-d] <device> [-i] <input> [-f] <refernece> [-e] [-n] <val> [-s] <val> [-o] [-b]
```
### 2. 参数说明

|命令参数           |是否必须|带参数 |参数说明                                       |  
|:------------------|:------:|:-----:|:-------------------------------------------|  
|-h, --help         |        |       |输出命令提示。                                |  
|-p, --proto        |&radic; |&radic;|指定tnnproto模型描述文件。                   |   
|-m, --model        |&radic; |&radic;|指定tnnmodel模型参数文件。                   |  
|-d, --device       |&radic; |&radic;|指定模型执行的平台，如OPENCL，ARM，METAL，CUDA，HUAWEI_NPU等。    |  
|-i, --input        |        |&radic;|指定输入文件。目前支持格式为：<br>&bull; 文本文件（文件后缀为.txt）, 格式与模型转换工具导出的输入格式一致。<br>&bull; 常用图片格式文件（文件后缀为 .jpg .jpeg .png .bmp）<br>如果不指定，则会使用 (-1, 1) 随机输入|  
|-f, --ref          |        |&radic;|采用指定输出进行结果对比。目前支持格式为：<br>&bull; 文本文件（文件后缀为.txt），格式与模型转换工具导出的输出格式一致。|  
|-e, --end          |        |       |仅校验模型的最终输出。                         |  
|-n, --bias         |        |&radic;|预处理，仅对输入为图片时有效。对输入数据各通道进行bias操作，参数格式为：0.0,0.0,0.0|  
|-s, --scale        |        |&radic;|预处理，仅对输入为图片时有效。对输入数据各通道进行scale操作，参数格式为：1.0,1.0,1.0|  
|-o, --output       |        |       |是否保存最终的输出。                           |  
|-b, --batch        |        |       |验证多batch情况下，每个batch结果是否正确。（还未开发完成） |  

## 四、执行脚本
### 1. Android
#### 1.1 模型准备
将待校验的模型的tnnproto和tnnmodel文件拷贝进`<path_to_tnn>/platforms/android/models`，并改名为`test.tnnproto`和`test.tnnmodel`
#### 1.2 执行脚本
```
cd <path_to_tnn>/platforms/android/
./model_check_android.sh -c -m <tnnproto> -p
```
### 2. Linux
#### 2.1. 编译脚本
```
cd <path_to_tnn>/platforms/linux/
./build_model_check.sh -c
```
#### 2.2. 执行命令
```
<path_to_tnn>/platforms/linux/build/model_check -p <path_to_tnnproto> -m <path_to_tnnmodel> -d <DEVICE>
```

## 五、工具限制
* 目前只支持fp32的模型校验；
* 对于逐层校验，只针对fp32精度下的结果进行校验；对于最后结果校验，使用Auto精度进行校验。
