# FAQ

[中文版本](../cn/faq.md)

## I. compilation questions

### Compilation environment requirements:
    General:  
        cmake >= 3.1  
        gcc >= 4.8  
        NDK >= r14b  
    Model conversion:  
        python >= 3.5  
        onnxruntime >= 1.1  
        onnx-simplifier >= 0.2.4  
        protobuf >= 3.0  
 
### ARMv8.2 compilation error
To support ARMv8.2 compilation, the ndk version must be at least r18b

### Windows CUDA Compilation
#### CUDA Version
  - CUDA 10.2 may produce the error not compatible with Visual Studio. If there is an error `cuda_toolset not found`, please reinstall CUDA and Visual Studio.
  - If there are multiple versions of CUDA installed, use `cmake -T` to choose the version you want to build. For example `cmake -Tcuda=10.2`

## II. Model conversion questions

### How to support tensorflow, caffe, mxnet models?
* We support the popular machine-learning training frameworks through intermediate onnx format, and the open source community provides handful tools for converting these frameworks to onnx
* [tensorflow2onnx](https://github.com/onnx/tensorflow-onnx): typical usage: python -m tf2onnx.convert --inputs-as-nchw [input tensor]: 0 --graphdef [input file].pb --inputs [input tensor]: 0 --outputs [output tensor]: 0 --opset 11 --output [output file].onnx
* [caffe2onnx](./user/caffe2tnn_en.md)
* [Mxnet: export onnx model](https://mxnet.apache.org/api/python/docs/tutorials/deploy/export/onnx.html)
* [Pytorch: EXPORTING A MODEL FROM PYTORCH TO ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

### Model alignment troubleshooting
* [Model alignment troubleshooting](./model_align_en.md) 

## III. Runtime questions

### Does it support running on PC
TNN supports compiling and running on linux and windows

### How to run bfp16 code
TNNTest's operating parameter -pr is set to LOW

### How to convert cv::Mat to TNN::Mat
```cpp
cv::Mat cv_mat;
MatType mat_type = N8UC4; // if cv_mat.channels() == 3, then mat_type = N8UC3.
DimsVector dims = {1, cv_mat.channels(), cv_mat.rows, cv_mat.cols};
auto tnn_mat = new TNN::Mat(DeviceType, mat_type, dims, (void *)cv_mat.ptr);
```

### Windows CUDA Error
  - Windows CUDA 10.2 error: `C:\source\rtSafe\cublas\cublasLtWrapper.cpp (279) - Assertion Error in nvinfer1::CublasLtWrapper::getCublasLtHeuristic: 0 (cublasStatus == CUBLAS_STATUS_SUCCESS)`
    Solution: Download [Patch]("https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal") on NVIDIA and install.
    
### Introduction to common error codes.
Status call the description() interface to get more error information description.  

0x1002(4098): Model parsing error. Check to make sure ModelConfig configures the file content instead of the file path.  

0x6005(24581): The model weights information is missing. The TNN benchmark can only use the proto file, because TNN_BENCHMARK_MODE is turned on, and the weights are automatically generated, which is only used to evaluate the speed.  

0x2000(8192): Error message not support model type. To check the Android static library integration link, you need to add -Wl,--whole-archive tnn -Wl,--no-whole-archive, and the iOS library integration link needs to add force_load.  

0x9000(36864): The device type is not supported. (1) Make sure that the relevant device type compilation options are turned on. (2) Android static library integration links need to add -Wl, --whole-archive tnn -Wl, --no-whole-archive, and iOS library integration links need to add force_load.  

## IV. NPU questions

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

### NPU Restriction
* If the NPU is of the version below 100.320.xxxxxxx
  ERROR: npu is installed but is below 100.320.xxx.xxx
* If the phone does not belong to Huawei or ROM version is too low：
  ERROR: GetRomVersion(ROM): npu is not installed or rom version is too low
  
### How to update the latest ROM version to support NPU?
* Go to Settings > System and Update > Software Update
    
### RKNPU Compilation Prerequisite:
1. Make directory named by `rknpu` under `<path_to_tnn>/third_party` and enter `rknpu`, then execute: `git clone https://github.com/airockchip/rknpu_ddk.git`
2. Add `-DTNN_RK_NPU_ENABLE:BOOL=ON` to `<path_to_tnn>/scripts/build_aarch64_linux.sh` and execute it.


## V. Others questions

### How to get intermediate results of the model?
* Modify [blob_dump_utils.h] (source/tnn/utils/blob_dump_utils.h)
*   \#define DUMP_INPUT_BLOB 0-> #define DUMP_INPUT_BLOB 1, get the input of each layer
*   \#define DUMP_OUTPUT_BLOB 0-> #define DUMP_OUTPUT_BLOB 1, get the output of each layer
* Only for debugging

### How to get the time cost of each layer of the model?
* Please refer to profiling document [performance test](./development/profiling_en.md)

### Internet problem
```text
// Homebrew installation under mac
//https://zhuanlan.zhihu.com/p/59805070
//https://brew.sh/index_zh-cn
// Replace the installation script of the domestic mirror
```
