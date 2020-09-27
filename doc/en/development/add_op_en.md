# Add a New Operator

[中文版本](../../cn/development/add_op.md)

If the operator is not found in [Operator List](../user/support_en.md), you can add a new operator through the following steps.
* [Add operator parser](#1)
* [Add Layer implementation](#2)
* [Add LayerAcc implementation](#3)
* [Add unit test](#4)

## 1. Add Operator Parser <span id = "1"> </span>
### 1.1 Add operator parameters

* Add LayerType
(1) Modify the file `<path_to_TNN>/source/tnn/core/layer_type.h`, add a new operator in` LayerType`, the format is `LAYER_XXX`.
(2) Modify the file `<path_to_TNN>/source/tnn/core/layer_type.cc` and add the operator name corresponding to the new operator value in` global_layer_type_map`, which needs to be consistent with the name of the layer in the proto file.

* Add LayerParam
If the new operator has other parameters besides the input and output blobs in proto, you need to add LayerParam: modify the file `<path_to_TNN>/source/tnn/interpreter/layer_param.h`, add a structure similar to` ConvLayerParam`, inherit from `LayerParam`
```
 struct ConvLayerParam: public LayerParam {
     int pad_type = -1;
     // input channels of blob, devide by group
     int input_channel = 0;
     // the total output channels of blob, not devide by group
     int output_channel = 0;
     // [w_begin w_end h_begin h_end d_begin d_end]
     std :: vector <int> pads;
     // order [w h d]
     std :: vector <int> kernels;
     // order [w h d]
     std :: vector <int> strides;
     // order [w h d]
     std :: vector <int> dialations;
     int group = 1;
     int bias = 0;
     int activation_type = ActivationType_None;
 };
```


* Add LayerResource
If the new operator has parameters that need to be saved in the model, you need to add LayerResource, modify the file `<path_to_TNN>/source/tnn/interpreter/layer_resource.h`, add a structure similar to` ConvLayerResource`, inherited from `LayerResource`
```
 struct ConvLayerResource: public LayerResource {
     // conv layer filter format
     ConvLayerFilterFormat filter_format = OIHW;

     // conv layer handle
     // NOTE: for deconv, the weight's default format is [n] [i] [o] [h] [w]
     RawBuffer filter_handle;

     // bias handle
     RawBuffer bias_handle;

     // extra scale handle for different precision
     RawBuffer scale_handle;
 };
```

### 1.2 Add LayerInterpreter
If the new operator adds LayerParam or LayerResource, you need to add the corresponding `LayerInterpreter`. Add the corresponding implementation in the folder `<path_to_TNN>/source/tnn/interpreter/tnn/layer_interpreter`.
(1) Declare the Interpreter of the new operator through `DECLARE_LAYER_INTERPRETER ()`;
(2) Register the Interpreter of the new operator through `REGISTER_LAYER_INTERPRETER ()`;
(3) Implement the following interface:
* `InterpretProto ()` -- Parsing the new operator's LayerParam
* `InterpretResource ()` -- Parsing the new operator's LayerResource
* `SaveProto ()` --  Save the new operator's LayerParam
* `SaveResource ()` -- Save the new operator's LayerResource

## 2. Add Layer Implementation <span id = "2"> </span>
Add the corresponding layer implementation under the folder `<path_to_TNN>/source/tnn/layer`.
(1) `DECLARE_LAYER ()` declares the Layer implementation of the new operator;
(2) `REGISTER_LAYER ()` register Layer implementation for new operators;
(3) Implement the following interface:
* `InferOutputDataType ()` -- set the data type of the corresponding layer's output blob
* `InferOutputShape ()` -- calculate the size of the output blob of the corresponding layer

## 3. Add LayerAcc Implementation <span id = "3"> </span>
Each new operator needs to implement the LayerAcc for the corresponding device.
### 3.1 CPU platform
Add the LayerAcc implementation of the corresponding operator in the folder `<path_to_TNN>/source/tnn/device/cpu/acc`.
(1) `DECLARE_CPU_ACC ()` declares the LayerAcc implementation of the new operator;
(2) `REGISTER_CPU_ACC ()` register LayerAcc implementation for the new operators;
(3) Implement the following interface:
* `Forward ()` -- CPU implementation of the new operator;
  
### 3.2 ARM platform
Add the LayerAcc implementation of the corresponding operator in the folder `<path_to_TNN>/source/tnn/device/arm/acc`.
(1) Declare the LayerAcc implementation of the new operator, if there are no special parameters, you can directly use the `DECLARE_ARM_ACC ()` declaration;
(2) `REGISTER_ARM_ACC ()` register LayerAcc implementation for the new operator;
(3) Implement the following interface:
* `Init ()` -- process LayerParam and LayerResource;
* `Reshape ()` -- implement logic when the input blob size changes;
* `Forward ()` -- ARM implementation of the new operator;


### 3.3 OpenCL platform
Add the LayerAcc implementation of the corresponding operator in the folder `<path_to_TNN>/source/tnn/device/opencl/acc`.
(1) Declare the LayerAcc implementation of the new operator. If there are no special parameters, you can directly use the `DECLARE_OPENCL_ACC ()` declaration;
(2) `REGISTER_OPENCL_ACC ()` register LayerAcc implementation for new operators;
(3) Implement the following interface:
* `Init ()` -- process LayerParam and LayerResource to create OpenCL kernel;
* `Reshape ()` -- to implement the logic when the input blob size changes, for OpenCL, call SegArgs here to set the kernel parameters;
* `Forward ()` -- implement OpenCL kernel;

(4) To implement OpenCL kernel, add the corresponding kernel file in the directory `<path_to_TNN>/source/tnn/device/opencl/cl`, with .cl as the suffix. After that, you need to execute the script:
 ```
 python opencl_codegen.py
 ```

### 3.4 Metal platform
Add the LayerAcc implementation of the corresponding operator in the folder `<path_to_TNN>/source/tnn/device/metal/acc`.
(1) Declare the LayerAcc implementation of the new operator, if there are no special parameters, you can directly use the `DECLARE_METAL_ACC ()` declaration;
(2) `REGISTER_METAL_ACC ()` register LayerAcc implementation for the new operator;
(3) Implement the following interface:
* `Init ()`
* `Reshape ()`
* `Forward ()`

### 3.5 Huawei NPU platform
In folder`<path_to_TNN>/source/tnn/device/huawei_npu/convert`, add  the LayerConvert implementation of the corresponding operator.
（1）Declare the LayerConvert implementation of the new operator，if no other input weights，Use`DECLARE_NPU_LAYER` to declare；  
（2）`REGISTER_NPU_LAYER` register LayerConvert implementation of new Operator；  
（3）Implement the following function：   
* `Convert()` -- use IR to convert the tnn operator；  

## 4. Add Unit Test <span id = "4"> </span>
Add the unit test file of the corresponding layer in the folder `<path_to_TNN>/test/unit_test/layer_test`.
