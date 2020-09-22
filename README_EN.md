[中文版本](README.md)
<div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/TNN.png"/>

## Introduction

TNN is a high-performance and lightweight inference framework for mobile devices. It provides lots of advanced features such as cross-platform, model-compression, and code-pruning. TNN, inspired by mainstream open-source industry frameworks, integrates and leverages Youtu Lab's Rapidnet, ncnn framework. It also combines the efforts of the deep-learning framework Oteam from all departments(PCG, TEG, IEG) to create an enterprise-level mobile inference engine.
At present, TNN has been launched to support various products in Youtu Lab and Guangying Studio.

## Effect Example
|    Face Detection(blazeface)     |   Object Detection(yolov5s)       |   Face Alignment<br>(from Tencent Youtu Lab)      |
|:---------|:-----------|:---------|
|[![blazeface](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/face_detection.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/blazeface) <br>iOS ✅  Android ✅ <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/blazeface)|[![yolov5s](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/object-detection.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/yolov5) <br> iOS ✅ Android ✅ <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/yolov5)   |[![youtu_facealign](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/face_alignment.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/youtu_face_alignment) <br> iOS ✅ Android ✅ <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/youtu_face_alignment)|

## Quick Start

It is very simple to use TNN. If you have a trained model, the model can be deployed on the target platform through three steps.
1. Convert the trained model into a TNN model. We provide a wealth of tools to help you complete this step, whether you are using Tensorflow, Pytorch, or Caffe, you can easily complete the conversion.
Detailed hands-on tutorials can be found here [How to Create a TNN Model](doc/en/user/convert_en.md).

2. When you have finished converting the model, the second step is to compile the TNN engine of the target platform. You can choose among different acceleration solutions such as ARM/OpenCL/Metal/NPU according to the hardware support.
   For these platforms, TNN provides convenient one-click scripts to compile. For detailed steps, please refer to [How to Compile TNN](doc/en/user/compile_en.md).

3. The final step is to use the compiled TNN engine for inference. You can make program calls to TNN inside your application. We provide a rich and detailed demo as a reference to help you complete.
    * [Run an iOS Demo](doc/en/user/demo_en.md)
    * [Run an Android Demo](doc/en/user/demo_en.md)

## Technical Solutions

TNN is a high-performance and lightweight inference framework for mobile devices. It provides lots of advanced features such as cross-platform, model-compression, and code-pruning. TNN, inspired by mainstream open-source industry frameworks, integrates and leverages Youtu Lab's Rapidnet, ncnn framework. It also combines the efforts of the deep-learning framework Oteam from all departments(PCG, TEG, IEG) to create an enterprise-level mobile inference engine.
At present, TNN has been launched in various major businesses, and its following characteristics have been widely praised.

* Computation optimization
    * The backend operators are primely optimized to make the best use of computing power in different architectures, regarding instruction issue, throughput, delay, cache bandwidth, cache delay, registers, etc..
    * The TNN performance on mainstream hardware platforms (CPU: ARMv7, ARMv8, GPU: Mali, Adreno, Apple) has been greatly tuned and improved.
    * The convolution function is implemented by various algorithms such as Winograd, Tile-GEMM, Direct Conv, etc., to ensure efficiency under different parameters and sizes.
    * Op fusion: TNN can do offline analysis of network graph, fuse multiple simple operations and reduce overhead such as redundant memory access and kernel startup cost.

* Low precision computation acceleration
    * TNN supports INT8/FP16 mode, reduces model size & memory consumption, and utilizes specific hardware low-precision instructions to accelerate calculations.
    * TNN supports INT8 WINOGRAD algorithm, (input 6bit), further reduces the model calculation complexity without sacrificing the accuracy.
    * TNN supports mixed-precision data in one model, speeding up the model's calculation speed while preserving its accuracy.

* Memory optimization
    * Efficient "memory pool" implementation: Based on a full network DAG analysis, the implementation reuses memory between non-dependent nodes which reduces memory cost by 90%.
    * Cross-model memory reduces: This supports external real-time design for network memory so that multiple models can share mutual memory.

* The performance of mainstream models on TNN: v0.1 2020.05.29

    * Kirin970：

        | model                     | cpu time(single thread, ms) | gpu time(ms) | npu time(ms)
        |---------------------------|--------------|--------------|---------------|
        | Mobilenet_v1              | 88           |   12         |       4.9     |
        | Mobilenet_v1_int8         | 55           |              |               |
        | Mobilenet_v2              | 58           |   11         |       8.0     |
        | Mobilenet_v2_int8         | 41           |              |               |
        | squeezenet_v1.0           | 127          |   20         |       5.1     |
        | squeezenet_v1.0_int8      | 82           |              |               |

    * Snapdragon 835：

        | model                     | cpu time(single thread, ms) | gpu time(ms) |
        |---------------------------|--------------|--------------|
        | Mobilenet_v1              | 94           |   16         |
        | Mobilenet_v1_int8         | 62           |              |
        | Mobilenet_v2              | 61           |   14         |
        | Mobilenet_v2_int8         | 47           |              |
        | squeezenet_v1.0           | 122          |   28         |
        | squeezenet_v1.0_int8      | 93           |              |

    * Snapdragon 845：

        | model                     | cpu time(single thread, ms) | gpu time(ms) |
        |---------------------------|--------------|--------------|
        | Mobilenet_v1              | 60           |   10         |
        | Mobilenet_v1_int8         | 37           |              |
        | Mobilenet_v2              | 39           |   8          |
        | Mobilenet_v2_int8         | 28           |              |
        | squeezenet_v1.0           | 74           |   14         |
        | squeezenet_v1.0_int8      | 56           |              |


* TNN architecture diagram：

   <div><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/imgs/tnn_architect.jpg"/>

* TNN supports TensorFlow, Pytorch, MxNet, Caffe, and other training frameworks through ONNX, leveraging the continuous improvement of the ONNX open-source society.
  Currently, TNN supports 55 ONNX operators and will be developed to cover 80 operators shortly, consisting of most of the mainstream CNN operators needed.
* TNN runs on mainstream operating systems (Android, iOS, embedded Linux), and is compatible with ARM CPU, GPU hardware platform (Da Vinci NPU will be supported soon)
* TNN is constructed through Modular Design, which abstracts and isolates components such as model analysis, graph construction, graph optimization, low-level hardware adaptation, and high-performance kernel.
   It uses "Factory Mode" to register and build devices, that tries to minimize the cost of supporting more hardware and acceleration solutions.
* TNN's running time does not rely on any third-party libraries. The size of the CPU dynamic library is only around 400KB, and it provides basic image conversion operations, which are light-weight and convenient. TNN uses unified models and interfaces across platforms and can switch easily by configuring just one single parameter.

## Learn About TNN Abilities
* [Operator Support](doc/en/user/support_en.md)
* [Model Support](doc/en/user/support_en.md)
* [Device Support](doc/en/user/support_en.md)
* [Profiling](doc/en/development/profiling_en.md)

## Manual
* [Compile TNN](doc/en/user/compile_en.md)
* [Tools]()
    * [Create a TNN Model](doc/en/user/convert_en.md)
    * [Model Quantization](doc/en/user/quantization_en.md)
    * [Model Visualization Netron](https://lutzroeder.github.io/netron/)
    * [Performance Analysis](doc/en/development/profiling_en.md)
    * [Model Alignment](doc/en/development/model_check_en.md)

## API Document
* [API call](doc/en/user/api_en.md)

## Contribute to TNN
* [Development Basics](doc/en/development/contributing_en.md)
* [Detailed Architecture](doc/en/development/architecture_en.md)
* [Add a New Operator](doc/en/development/add_op_en.md)
* [Unit Test](doc/en/development/unit_test_en.md)

## Roadmap
* [Road map](doc/cn/user/roadmap.md)

## Acknowledgement
TNN referenced the following projects：

* [ncnn](https://github.com/Tencent/ncnn)

* [mace](https://github.com/XiaoMi/mace.git)

* [MNN](https://github.com/alibaba/MNN)

* [caffe-onnx](https://github.com/htshinichi/caffe-onnx)

* [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)

* [onnx](https://github.com/onnx/onnx)

## License
* [BSD 3 Clause](LICENSE)

## FAQ
* [FAQ](doc/en/faq_en.md)

## Join Us

* Everyone is welcome to participate to build the best mobile inference framework in the industry.

* Technical Discussion QQ Group: 913940506 Answer: TNN

* Scan the QR code to join the TNN discussion group：
<div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/TNN-QQ.png"/>

## FAQ
* [Frequently asked questions](/doc/en/faq_en.md)
