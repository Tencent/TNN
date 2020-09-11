# FAQ 常见问题

[English Version](../en/faq_en.md)

## 编译环境要求：
    general:
        cmake >= 3.1
        gcc >= 4.8
        NDK >= r14b
    模型转换：
        python >= 3.5
        onnxruntime>=1.1
        onnx-simplifier>=0.2.4
        protobuf >= 3.0

## 如何创建华为NPU编译环境? 
选项1: 
  在 <TNN_PROJECT>/thrid_party/huawei_npu/ 下运行 ./download_ddk.sh 脚本下载最新版的ddk。
  

选项2：
1. 到华为开发者联盟下载DDK[https://developer.huawei.com/consumer/cn/doc/overview/HUAWEI_HiAI]
2. 解压缩
3. 进入到下载文件夹下的`ddk/ai_ddk_lib`目录
4. 在`<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/`下创建`armeabi-v7a`文件夹， 并将ai_ddk_lib目录下的lib文件夹中所有文件复制到 `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/armeabi-v7a`
5. 在`<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/`下创建`arm64-v8a`文件夹，并将ai_ddk_lib目录下的lib64文件夹中所有文件复制到 `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/arm64-v8a`
6. 将ai_ddk_lib目录下include`文件夹`复制到 `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/`目录下

### 最终 `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/`文件结构应该如下：

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

## NPU版本限制：
* 如果获取手机的ROM在100.320.xxx.xxx以下
  报错
  ERROR: npu is installed but is below 100.320.xxx.xxx
* 如果没有npu或是非华为手机 ：
  报错 
  ERROR: GetRomVersion(ROM): npu is not installed or rom version is too low
  
## 如何更新到最新的ROM去支持NPU？ 
* 到 设置 >> 系统和更新 >> 软件更新中检查最新的ROM版本并更新。
        
## 模型支持：

### 如何支持tensorflow, caffe, mxnet模型？
* 我们统一通过onnx中间格式支持各大训练框架，开源社区维护有很好的各大框架转换为onnx的工具
* [tensorflow2onnx](https://github.com/onnx/tensorflow-onnx): typical usage: python -m tf2onnx.convert --inputs-as-nchw [输入tensor]:0   --graphdef [输入文件].pb  --inputs [输入tensor]:0  --outputs [输出tensor]:0  --opset 11 --output [输出文件].onnx
* [caffe2onnx](./user/caffe2tnn.md)
* [Mxnet: export onnx model](https://mxnet.apache.org/api/python/docs/tutorials/deploy/export/onnx.html)
* [Pytorch: EXPORTING A MODEL FROM PYTORCH TO ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

## 如何确定结果是否正确？
* 参照[模型测试文档](./user/test.md)


## 如何获取模型中间结果？
* 修改项目目录下 /source/tnn/utils/blob_dump_utils.h 中
*    \#define DUMP_INPUT_BLOB 0  --> #define DUMP_INPUT_BLOB 1，获取每层输入
*    \#define DUMP_OUTPUT_BLOB 0 --> #define DUMP_OUTPUT_BLOB 1，获取每层输出
* 仅作为调试使用

## 如何获取模型各个layer耗时？
* 参考profiling文档[性能测试](./development/profiling.md)

## 网络问题
```text
//mac下homebrew安装
//https://zhuanlan.zhihu.com/p/59805070
//https://brew.sh/index_zh-cn
//替换国内镜像的安装脚本
```
