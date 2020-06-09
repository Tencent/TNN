# FAQ

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
