# FAQ

[中文版本](../cn/faq.md)

## Compilation environment requirements:
    General:
        cmake >= 3.1
        gcc >= 4.8
        NDK >= r14b
    Model conversion:
        python >= 3.5
        onnxruntime >= 1.1
        onnx-simplifier >= 0.2.4
        protobuf >= 3.0
        

## Huawei NPU Compilation Prerequisite:
You need the ddk to support where you could obain by 
Either 
Option 1 :
Go to <path_to_tnn>/thrid_party/huawei_npu/, use ./download_ddk.sh to download the ddk.
````
cd <path_to_tnn>/thrid_party/huawei_npu/
./download_ddk.sh 
````
Option 2 :
1. Downlaod DDK from the following path [https://developer.huawei.com/consumer/cn/doc/overview/HUAWEI_HiAI]
2. unzip 
3. Go to the `ddk/ai_ddk_lib` directory under the downloaded folder 
4. Make directory named by `armeabi-v7a`under  `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/`, and opy all files under the `ddk/ai_ddk_lib/lib` directory to `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/armeabi-v7a`
5.  Make directory named by `arm64-v8a`under  `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/`,and copy all files under the `ddk/ai_ddk_lib/lib64` directory to  `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/arm64-v8a`
6. Copy the `include` directory to `<path_to_tnn>>/third_party/huawei_npu/hiai_ddk_latest/`
7. The structure of the `<path_to_tnn>/third_party/huawei_npu/hiai_ddk_latest/` should be like：

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

## NPU Restriction
* If the NPU is of the version below 100.320.xxxxxxx
  ERROR: npu is installed but is below 100.320.xxx.xxx
* If the phone does not belong to Huawei or ROM version is too low：
  ERROR: GetRomVersion(ROM): npu is not installed or rom version is too low
  
## How to update the latest ROM version to support NPU?
* Go to Settings > System and Update > Software Update
    
## Model support:

### How to support tensorflow, caffe, mxnet models?
* We support the popular machine-learning training frameworks through intermediate onnx format, and the open source community provides handful tools for converting these frameworks to onnx
* [tensorflow2onnx](https://github.com/onnx/tensorflow-onnx): typical usage: python -m tf2onnx.convert --inputs-as-nchw [input tensor]: 0 --graphdef [input file].pb --inputs [input tensor]: 0 --outputs [output tensor]: 0 --opset 11 --output [output file].onnx
* [caffe2onnx](./user/caffe2tnn_en.md)
* [Mxnet: export onnx model](https://mxnet.apache.org/api/python/docs/tutorials/deploy/export/onnx.html)
* [Pytorch: EXPORTING A MODEL FROM PYTORCH TO ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

## How to check if the result is correct?
* Refer to [Model Test Document](./user/test_en.md)


## How to get intermediate results of the model?
* Modify [blob_dump_utils.h] (source/tnn/utils/blob_dump_utils.h)
*   \#define DUMP_INPUT_BLOB 0-> #define DUMP_INPUT_BLOB 1, get the input of each layer
*   \#define DUMP_OUTPUT_BLOB 0-> #define DUMP_OUTPUT_BLOB 1, get the output of each layer
* Only for debugging

## How to get the time cost of each layer of the model?
* Please refer to profiling document [performance test](./development/profiling_en.md)

## Internet problem
```text
// Homebrew installation under mac
//https://zhuanlan.zhihu.com/p/59805070
//https://brew.sh/index_zh-cn
// Replace the installation script of the domestic mirror
```
