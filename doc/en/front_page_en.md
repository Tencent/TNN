[中文版本](/doc/cn/front_page.md)
<div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/TNN.png"/>

## Get Started

It is very simple to use TNN. If you have a trained model, the model can be deployed on the target platform through three steps.
1. Convert the trained model into a TNN model. We provide a wealth of tools to help you complete this step, whether you are using Tensorflow, Pytorch, or Caffe, you can easily complete the conversion.
Detailed hands-on tutorials can be found here [How to Create TNN Models](doc/en/user/convert_en.md).

2. When you have finished converting the model, the second step is to compile the TNN engine of the target platform. You can choose among different acceleration solutions such as CPU / ARM / OpenCL / Metal according to the hardware support.
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

* Performance comparison among mainstream models: TNN outperforms other mainstream open-source mobile high-performance frameworks.
    * Kirin970：

    <div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/imgs/970.jpg"/>

    * Snapdragon 835：

    <div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/imgs/835.jpg"/>


* TNN architecture diagram：

   <div align=left><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/imgs/tnn_architect.jpg" width="512" alt=TNN架构 />

* TNN supports TensorFlow, Pytorch, MxNet, Caffe, and other training frameworks through ONNX, leveraging the continuous improvement of the ONNX open-source society.
  Currently TNN supports 55 ONNX operators, and will be developed to cover 80 operators shortly, consisting of most of the mainstream CNN operators needed.
* TNN runs on mainstream operating systems (Android, iOS, embedded Linux, Windows), and is compatible with ARM CPU, GPU hardware platform (Da Vinci NPU will be supported soon)
* TNN is constructed through Modular Design, which abstracts and isolates components such as model analysis, graph construction, graph optimization, low-level hardware adaptation, and high-performance kernel.
   It uses "Factory Mode" to register and build devices, that tries to minimize the cost of supporting more hardware and acceleration solutions.
* TNN's running time does not rely on any third-party libraries. The size of the CPU dynamic library is only around 400KB, and it provides basic image conversion operations, which are light-weight and convenient. TNN uses unified models and interfaces across platforms and can switch easily by configuring just one single parameter.

## Learn about the TNN Spec
* [Operator Support](./user/support_en.md)
* [Model Support](./user/support_en.md)
* [Device Support](./user/support_en.md)
* [Profiling](./development/profiling_en.md)

## Manual
* [Compile TNN](./user/compile_en.md)
* [Tools]()
    * [Create TNN Models](./user/convert_en.md)
    * [Model Quantization](./user/quantization_en.md)
    * [Model Visualization](.user/visual_en.md)
    * [Performance Analysis](./development/profiling_en.md)
    * [Model Alignment](./development/model_check_en.md)

## API Document
* [API call](./user/api_en.md)

## Integration & Development
* [Development basics](./development/contributing_en.md)
* [Detailed architecture](./development/architecture_en.md)
* [Add a New Operator](.development/add_op_en.md)
* [Unit test](./development/unit_test_en.md)

## Roadmap
* [Road map]()

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