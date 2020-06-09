# FAQ 常见问题

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
