# 新增OP  

[English Version](../../en/development/add_op_en.md)

如果需要的算子不在[算子列表](../user/support.md)中，则需要通过以下步骤添加新的算子。  
* [添加算子解析](#1)  
* [添加Layer实现](#2)  
* [添加LayerAcc实现](#3)  
* [添加单元测试](#4)  

## 1. 添加算子解析 <span id = "1"></span>
### 1.1 添加算子参数 

* 添加LayerType  
（1）修改文件 `<path_to_TNN>/source/tnn/core/layer_type.h`，在`LayerType`中添加新算子的枚举，格式为`LAYER_XXX`。  
（2）修改文件 `<path_to_TNN>/source/tnn/core/layer_type.cc`，在`global_layer_type_map`中添加新算子枚举值对应的算子名称，此名称与proto文件中层的名称一致。  

* 添加LayerParam    
如果新算子在proto里除了输入输出blob，还有其他参数，则需要添加LayerParam，修改文件 `<path_to_TNN>/source/tnn/interpreter/layer_param.h`，添加类似`ConvLayerParam`的结构，继承于`LayerParam`

```cpp
 struct ConvLayerParam : public LayerParam {
     int pad_type = -1;
     // input channels of blob, divide by group
     int input_channel = 0;
     // the total output channels of blob, not devide by group
     int output_channel = 0;
     //[w_begin w_end h_begin h_end d_begin d_end]
     std::vector<int> pads;
     // order [w h d]
     std::vector<int> kernels;
     // order [w h d]
     std::vector<int> strides;
     // order [w h d]
     std::vector<int> dialations;
     int group           = 1;
     int bias            = 0;
     int activation_type = ActivationType_None;
 };
```

* 添加LayerResource    
如果新算子有需要保存到model里的参数，则需要添加LayerResource，修改文件 `<path_to_TNN>/source/tnn/interpreter/layer_resource.h`，添加类似`ConvLayerResource`的结构，继承于`LayerResource`

```cpp
 struct ConvLayerResource : public LayerResource {
     // conv layer filter format
     ConvLayerFilterFormat filter_format = OIHW;

     // conv layer handle
     // NOTE: for deconv, the weight's default format is  [n][i][o][h][w]
     RawBuffer filter_handle;

     // bias handle
     RawBuffer bias_handle;

     // extra scale handle for different precision
     RawBuffer scale_handle;
 };
```

### 1.2 添加LayerInterpreter 
如果新算子添加了LayerParam或者LayerResource，则需要添加对应的`LayerInterpreter`。在文件夹`<path_to_TNN>/source/tnn/interpreter/tnn/layer_interpreter`下添加对应的实现。  
(1）通过`DECLARE_LAYER_INTERPRETER()`声明新算子的Interpreter；  
(2）通过`REGISTER_LAYER_INTERPRETER()`注册新算子的Interpreter；  
(3）实现以下接口：  
* `InterpretProto()` -- 解析新算子的LayerParam  
* `InterpretResource()`  -- 解析新算子的LayerResource  
* `SaveProto()`  -- 保存新算子的LayerParam  
* `SaveResource()`  -- 保存新算子的LayerResource  

## 2. 添加Layer实现 <span id = "2"></span>  
在文件夹 `<path_to_TNN>/source/tnn/layer` 下添加对应layer的实现。   
(1）`DECLARE_LAYER()` 声明新算子的Layer实现；   
(2）`REGISTER_LAYER()` 注册新算子的Layer实现；     
(3）实现以下接口：   
* `InferOutputDataType()` -- 设置对应层输出Blob的数据类型  
* `InferOutputShape()` -- 计算对应层输出Blob的大小  

## 3. 添加LayerAcc实现 <span id = "3"></span>
每个新的算子都需要实现对应设备的LayerAcc。  
### 3.1 CPU平台  
在文件夹`<path_to_TNN>/source/tnn/device/cpu/acc`下添加对应算子的LayerAcc实现。  
(1）`DECLARE_CPU_ACC()` 声明新算子的LayerAcc实现；  
(2）`REGISTER_CPU_ACC()` 注册新算子的LayerAcc实现；  
(3）实现以下接口：  
* `Forward()` -- 新算子的cpu实现；  
  
### 3.2 ARM平台  
在文件夹`<path_to_TNN>/source/tnn/device/arm/acc`下添加对应算子的LayerAcc实现。    
(1）声明新算子的LayerAcc实现，如果没有特殊的参数，可以直接使用`DECLARE_ARM_ACC()`声明；  
(2）`REGISTER_ARM_ACC()` 注册新算子的LayerAcc实现；  
(3）实现以下接口：  
* `Init()` -- 对LayerParam和LayerResource进行处理；  
* `Reshape()` -- 实现在输入blob大小改变的情况下的逻辑；  
* `Forward()` -- 新算子的ARM实现；  

### 3.3 OpenCL平台  
在文件夹`<path_to_TNN>/source/tnn/device/opencl/acc`下添加对应算子的LayerAcc实现。  
(1）声明新算子的LayerAcc实现，如果没有特殊的参数，可以直接使用`DECLARE_OPENCL_ACC()`声明；  
(2）`REGISTER_OPENCL_ACC()` 注册新算子的LayerAcc实现；  
(3）实现以下接口：  
* `Init()` -- 对LayerParam和LayerResource进行处理，创建OpenCL的kernel；  
* `Reshape()` -- 实现在输入blob大小改变的情况下的逻辑，对于OpenCL，在此处调用SegArgs设置kernel参数；  
* `Forward()` -- 执行OpenCL的kernel；  

(4）实现OpenCL的kernel，在目录 `<path_to_TNN>/source/tnn/device/opencl/cl` 添加对应的kernel文件，以.cl为后缀。添加之后需要执行脚本:
 
 ``` python
 python opencl_codegen.py
 ```

### 3.4 Metal平台  
在文件夹`<path_to_TNN>/source/tnn/device/metal/acc`下添加对应算子的LayerAcc实现。
(1）声明新算子的LayerAcc实现，如果没有特殊的参数，可以直接使用`DECLARE_METAL_ACC()`声明；  
(2）`REGISTER_METAL_ACC()` 注册新算子的LayerAcc实现；  
(3）实现以下接口：  
* `Init()`  
* `Reshape()`    
* `Forward()`    

(4）实现Metal的kernel，在目录 `<path_to_TNN>/source/tnn/device/metal/acc` 添加对应的metal文件，以.metal为后缀。

### 3.5 NPU平台  
在文件夹`<path_to_TNN>/source/tnn/device/huawei_npu/convert`下添加对应算子的LayerConvert实现。  
(1）声明新算子的LayerConvert实现，如果没有其他权重input，可以直接使用`DECLARE_NPU_LAYER`声明；  
(2）`REGISTER_NPU_LAYER` 注册新算子的LayerConvert实现；  
(3）实现以下接口：   
* `Convert()` -- 使用ir翻译tnn模型算子；  

### 3.6 X86平台  

#### 3.6.1 openvino算子导入
在文件夹`<path_to_TNN>/source/tnn/network/openvino/layer_builder`下添加对应算子的OpenVINOLayerBuilder实现。  
(1）声明新算子的OpenVINOLayerBuilder实现，可以直接使用`DECLARE_OPENVINO_LAYER_BUILDER`声明；  
(2）`REGISTER_OPENVINO_LAYER_BUILDER` 注册新算子的LayerConvert实现；  
(3）实现以下接口：   
* `Build()` -- 将tnn的算子转换成ngraph的node；  
  
对于openvino不支持或者性能较差的算子可以注册custom op来替代openvino的op。
(1）在`<path_to_TNN>/source/tnn/network/openvino/custom_layer`下使用`DECLARE_CUSTOM_IMPLEMENTATION`和`REGISTER_CUSTOM_IMPLEMENTATION`进行声明和注册
(2）在`Build()`函数里使用已注册的custom op来构建ngraph的node

#### 3.6.2 kernel编写
在文件夹`<path_to_TNN>/source/tnn/device/x86/acc`下添加对应算子的LayerAcc实现。    
(1）声明新算子的LayerAcc实现，如果没有特殊的参数，可以直接使用`DECLARE_X86_ACC()`声明；  
(2）`REGISTER_X86_ACC()` 注册新算子的LayerAcc实现；  
(3）实现以下接口：  
* `Init()` -- 对LayerParam和LayerResource进行处理；  
* `Reshape()` -- 实现在输入blob大小改变的情况下的逻辑；  
* `Forward()` -- 新算子的X86实现；  

### 3.7 CUDA平台  

#### 3.7.1 tensorrt算子导入
在文件夹`<path_to_TNN>/source/tnn/network/tensorrt/layer_builder`下添加对应算子的TensorRTLayerBuilder实现。  
(1）声明新算子的TensorRTLayerBuilder实现，可以直接使用`DECLARE_TENSORRT_LAYER_BUILDER`声明；  
(2）`REGISTER_TENSORRT_LAYER_BUILDER` 注册新算子的TensorRTLayerBuilder实现；  
(3）实现以下接口：   
* `AddToNetwork()` -- 网络导入对应的tensorrt算子。  
  
对于tensorrt不支持或者性能较差的算子可以注册plugin op来替代tensorrt的op。
(1）在`<path_to_TNN>/source/tnn/network/tensorrt/layer_build`下使用`DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER`和`REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER`进行声明和注册
(2）实现以下接口：  
*  `supportsFormatCombination` -- 判断支持的数据类型和数据排布；  
*  `getPluginType` -- 自定义plugin type；  
*  `getOutputDataType` -- 设定输出data type； 
*  `AddToNetwork` -- 实现插件导入网络； 
*  `getOutputDimensions` -- 返回输出尺寸计算公式
*  `getPluginName` -- 自定义plugin name

#### 3.7.2 kernel编写
在文件夹`<path_to_TNN>/source/tnn/device/cuda/acc`下添加对应算子的LayerAcc实现。    
(1）声明新算子的LayerAcc实现，如果没有特殊的参数，可以直接使用`DECLARE_CUDA_ACC()`声明；  
(2）`REGISTER_CUDA_ACC()` 注册新算子的LayerAcc实现；  
(3）实现以下接口：  
* `Init()` -- 对LayerParam和LayerResource进行处理；  
* `Reshape()` -- 实现在输入blob大小改变的情况下的逻辑;  
* `Forward()` -- 新算子的CUDA实现;    

## 4. 添加单元测试 <span id = "4"></span>  
在文件夹 `<path_to_TNN>/test/unit_test/layer_test` 下添加对应层的单元测试文件。
