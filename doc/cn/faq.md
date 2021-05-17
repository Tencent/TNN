# FAQ 常见问题

[English Version](../en/faq_en.md)

## 一、编译问题

### 编译环境要求：
    general:  
        cmake >= 3.1  
        gcc >= 4.8  
        NDK >= r14b  
    模型转换:  
        python >= 3.5  
        onnxruntime>=1.1  
        onnx-simplifier>=0.2.4  
        protobuf >= 3.0  

### ARMv8.2编译报错
若要支持ARMv8.2编译，ndk版本版本至少为r18b

### Windows CUDA 编译
#### CUDA 版本
  cuda 10.2 可能存在与 Visual Studio 安装不完整的问题，如出现 cuda_toolset not found ，请重新安装或升级 CUDA 及 Visual Studio。


        
## 二、模型转换问题

### 如何支持tensorflow, caffe, mxnet模型？
* 我们统一通过onnx中间格式支持各大训练框架，开源社区维护有很好的各大框架转换为onnx的工具  
* [tensorflow2onnx](https://github.com/onnx/tensorflow-onnx): typical usage: python -m tf2onnx.convert --inputs-as-nchw [输入tensor]:0   --graphdef [输入文件].pb  --inputs [输入tensor]:0  --outputs [输出tensor]:0  --opset 11 --output [输出文件].onnx  
* [caffe2onnx](./user/caffe2tnn.md)  
* [Mxnet: export onnx model](https://mxnet.apache.org/api/python/docs/tutorials/deploy/export/onnx.html)  
* [Pytorch: EXPORTING A MODEL FROM PYTORCH TO ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) 

### 模型对齐问题排查
* [模型对齐问题排查](./model_align.md) 

## 三、运行问题

### 是否支持可以在PC上运行
TNN支持在linux和windows上编译和运行

### 如何运行bfp16代码
TNNTest的运行参数-pr设为LOW

### cv::Mat如何转换成TNN::Mat
```cpp
cv::Mat cv_mat;
MatType mat_type = N8UC4; // if cv_mat.channels() == 3, then mat_type = N8UC3.
DimsVector dims = {1, cv_mat.channels(), cv_mat.rows, cv_mat.cols};
auto tnn_mat = new TNN::Mat(DeviceType, mat_type, dims, (void *)cv_mat.ptr);
```

### 常见错误码介绍. 
Status调用description()接口可获取更多错误信息描述。

0x1002(4098): 模型解析错误。检查确保ModelConfig配置的为文件内容而非文件路径。  

0x6005(24581): 模型weights信息缺失。TNN的benchmark可以只用proto文件，是因为开启了TNN_BENCHMARK_MODE，weights自动生成，仅用来评估速度。  

0x2000(8192): 错误信息not support model type。检查Android静态库集成链接需添加-Wl,--whole-archive tnn -Wl,--no-whole-archive，iOS库集成链接需要添加force_load。  

0x9000(36864): device type类型不支持。（1）确保相关device type编译选项已开启 （2）Android静态库集成链接需添加-Wl,--whole-archive tnn -Wl,--no-whole-archive，iOS库集成链接需要添加force_load。  

##  四、NPU相关问题

### 如何创建华为NPU编译环境? 
选项1: 
  在 <TNN_PROJECT>/thrid_party/huawei_npu/ 下运行 ./download_ddk.sh 脚本下载最新版的ddk。
  

选项2：
1. 到华为开发者联盟下载DDK[https://developer.huawei.com/consumer/cn/doc/overview/HUAWEI_HiAI]
2. 解压缩
3. 进入到下载文件夹下的`ddk/ai_ddk_lib`目录
4. 在`<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/`下创建`armeabi-v7a`文件夹， 并将ai_ddk_lib目录下的lib文件夹中所有文件复制到 `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/armeabi-v7a`
5. 在`<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/`下创建`arm64-v8a`文件夹，并将ai_ddk_lib目录下的lib64文件夹中所有文件复制到 `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/arm64-v8a`
6. 将ai_ddk_lib目录下include`文件夹`复制到 `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/`目录下

`<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/`文件结构应该如下：

```
hiai_ddk_latest
├── arm64-v8a 
│   ├── libcpucl.so 
│   ├── libhcl.so
│   ├── libhiai.so
│   ├── libhiai_ir.so
│   └── libhiai_ir_build.so
├── armeabi-v7a
│   ├── libcpucl.so
│   ├── libhcl.so
│   ├── libhiai.so
│   ├── libhiai_ir.so
│   └── libhiai_ir_build.so
└── include
    ├── HiAiAippPara.h
    ├── HiAiModelManagerService.h
    ├── HiAiModelManagerType.h
    ├── graph
    │   ├── attr_value.h
    │   ├── buffer.h
    │   ├── common
    │   │   └── secures\tl.h
    │   ├── debug
    │   │   └── ge_error_codes.h
    │   ├── detail
    │   │   └── attributes_holder.h
    │   ├── graph.h
    │   ├── model.h
    │   ├── op
    │   │   ├── all_ops.h
    │   │   ├── array_defs.h
    │   │   ├── const_defs.h
    │   │   ├── detection_defs.h
    │   │   ├── image_defs.h
    │   │   ├── math_defs.h
    │   │   ├── nn_defs.h
    │   │   └── random_defs.h
    │   ├── operator.h
    │   ├── operator_reg.h
    │   ├── tensor.h 
    │   └── types.h
    └── hiai_ir_build.h
```

### NPU版本限制：
* 如果获取手机的ROM在100.320.xxx.xxx以下
  报错
  ERROR: npu is installed but is below 100.320.xxx.xxx
* 如果没有npu或是非华为手机 ：
  报错 
  ERROR: GetRomVersion(ROM): npu is not installed or rom version is too low
  
### 如何更新到最新的ROM去支持NPU？ 
* 到 设置 >> 系统和更新 >> 软件更新中检查最新的ROM版本并更新。

### 如何创建RKNPU编译环境? 
1. 在`<path_to_tnn>/third_party`下创建rknpu文件夹并进入，然后执行： `git clone https://github.com/airockchip/rknpu_ddk.git`。
2. 在`<path_to_tnn>/scripts/build_aarch64_linux.sh`文件中加入`-DTNN_RK_NPU_ENABLE:BOOL=ON`选项并编译即可。


## 五、其他
### 如何获取模型中间结果？
* 修改项目目录下 /source/tnn/utils/blob_dump_utils.h 中
*    \#define DUMP_INPUT_BLOB 0  --> #define DUMP_INPUT_BLOB 1，获取每层输入
*    \#define DUMP_OUTPUT_BLOB 0 --> #define DUMP_OUTPUT_BLOB 1，获取每层输出
* 仅作为调试使用

### 七、如何获取模型各个layer耗时？
* 参考profiling文档[性能测试](./development/profiling.md)

### 网络问题
```text
//mac下homebrew安装
//https://zhuanlan.zhihu.com/p/59805070
//https://brew.sh/index_zh-cn
//替换国内镜像的安装脚本
```
