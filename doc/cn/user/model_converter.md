# 内部模型互转

[English Version](../../en/user/model_converter_en.md)

## 一、工具的作用
+ 在以下三种模型格式中相互转换：  
    + RapidnetV1：旧版本的RPN模型
    + TNN：目前开源版本模型
    + RapidnetV3：TNN的加密版模型  
+ 打印模型的基本信息，比如模型的版本号，模型的存储精度；

## 二、编译
编译model_converter工具需要将以下宏设置为ON。（编译方法参照[TNN编译文档](../user/compile.md))
* `TNN_MODEL_CONVERT_ENABLE`  

## 三、校验工具使用
### 1. 命令
```
./model_convert [-h] [-i] [-p] <proto_path> [-m] <model_path> [-v] <version>
```
### 2. 参数说明

|命令参数           |是否必须|带参数 |参数说明                                       |  
|:------------------|:------:|:-----:|:-------------------------------------------|  
|-h, --help         |        |       |输出命令提示。                               | 
|-i, --info         |        |       |输出模型的相关信息                           |   
|-p, --proto        |&radic; |&radic;|指定tnnproto模型描述文件。                   |   
|-m, --model        |&radic; |&radic;|指定tnnmodel模型参数文件。                   |  
|-v, --version      |        |&radic;|指定输出模型的版本。参数如下：<br>&bull; 0：RapidnetV1 <br>&bull; 1：TNN <br>&bull; 2：RapidnetV3（默认）|  

## 四、执行脚本
### 1. Linux
#### 1.1. 编译脚本
```
cd <path_to_tnn>/platforms/linux/
./build_model_convert.sh -c
```
#### 1.2. 执行命令
+ 模型互转  
```
<path_to_tnn>/platforms/linux/build/model_converter -p <path_to_tnnproto> -m <path_to_tnnmodel> -v <version>
```
+ 模型信息打印  

```
<path_to_tnn>/platforms/linux/build/model_converter -p <path_to_tnnproto> -m <path_to_tnnmodel> -i
```

## 五、工具限制
+ 指定的proto文件和model文件一定要是对应的，目前不支持校验两个文件是否一致；
