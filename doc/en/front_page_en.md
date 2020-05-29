[中文版本](/doc/cn/front_page.md)
# <div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/TNN.png" height="50%" width="50%"/>
## Get Started
Using TNN is very simple. If you have a trained model, the model can be deployed on the target platform through the following three steps in general.
1. The first step is to convert the trained model into a TNN model. We provide a bunch of tools to help you complete this step, whether you are using Tensorflow, Pytorch, or Caffe, you can easily complete the conversion.
Detailed hands-on tutorials can be found here [How to Convert Model](./user/convert_en.md).

2. When you have completed the model conversion, the second step is to compile the TNN engine of the target platform. You can choose the acceleration solution such as CPU/ARM/OpenCL/Metal according to the hardware support of your target platform. For these platforms, TNN provides one-click built scripts, which are very convenient to use. For detailed steps, please refer to [How to Compile TNN](./user/compile_en.md).

3. The final step is to use the compiled TNN engine for inference. You can embed the call to TNN in your application. We provide a detailed demo to show you how to do this.
    * [Start an iOS Demo from 0](./user/demo_en.md)
    * [Start an Android Demo from 0](./user/demo_en.md)

## Technical solutions
TNN is a high-performance and light-weight inference framework dedicated to mobile devices. it's cross-platform and has outstanding features such as model compression and code prunning. TNN is inspired by the industry's mainstream open source framework, integrates and leverages Youtu Lab's Rapidnet, ncnn framework. It combines the efforts of creating a company-level mobile inference engine by all departments(PCG, TEG, IEG) of the deep learning framework Oteam.
At present, TNN has been supporting various businesses, and its following characteristics have been widely praised.

#### Outstanding Performance
* Computation optimization
    * The backend operators are primely optimized to make the best use of computing power in different architectures, regarding instruction issue, throughput, delay, cache bandwidth, cache delay, registers, etc..
    * The TNN performance on mainstream hardware platforms (CPU: ARMv7, ARMv8, GPU: Mali, Adreno, Apple) has been greatly tuned and improved.
    * The convolution function is implemented by various algorithms such as Winograd, Tile-GEMM, Direct Conv, etc., to ensure efficiency under different parameters and sizes.
    * Op fusion: TNN can do an offline analysis of network graph, fuse multiple simple operations and reduce overhead such as redundant memory access and kernel startup cost.

* Low precision computation acceleration
    * TNN supports INT8/FP16 mode, reduces model size & memory consumption, and utilizes specific hardware low-precision instructions to accelerate calculations.
    * TNN supports INT8 WINOGRAD algorithm, (input 6bit), further reduces the model calculation complexity without sacrificing the accuracy.
    * TNN supports mixed-precision data in one model, speeding up the model's calculation speed while preserving its accuracy.

* Memory optimization
    * Efficient "memory pool" implementation: Based on a full network DAG analysis, the implementation reuses memory between non-dependent nodes which reduces memory cost by 90%.
    * Cross-model memory reduces: This supports external real-time design for network memory so that multiple models can share mutual memory.

* Performance comparison among mainstream models: TNN outperforms other mainstream open-source mobile high-performance frameworks.

    * Kirin970：

    <div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/imgs/970.jpg" width="512" alt=华为麒麟970平台 />

    * Snapdragon 835：

    <div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/imgs/835.jpg" width="512" alt=高通骁龙835平台 />

### Universal & Lightweight：

#### TNN architecture diagram：

   <div align=left><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/imgs/tnn_architect.jpg" width="512" alt=TNN架构 />

* TNN supports TensorFlow, Pytorch, MxNet, Caffe, and other training frameworks through ONNX, leveraging the continuous improvement of the ONNX open-source society.
  Currently TNN supports 55 ONNX operators, and will be developed to cover 80 operators shortly, consisting of most of the mainstream CNN operators needed.
* TNN runs on mainstream operating systems (Android, iOS, embedded Linux, Windows), and is compatible with ARM CPU, GPU hardware platform (Da Vinci NPU will be supported soon)
* TNN is constructed through Modular Design, which abstracts and isolates components such as model analysis, graph construction, graph optimization, low-level hardware adaptation, and high-performance kernel.
   It uses "Factory Mode" to register and build devices, that tries to minimize the cost of supporting more hardware and acceleration solutions.
* TNN's running time does not rely on any third-party libraries. The size of the CPU dynamic library is only around 400KB, and it provides basic image conversion operations, which are light-weight and convenient. TNN uses unified models and interfaces across platforms and can switch easily by configuring just one single parameter.


## Support and Test
* [Supported OP](./user/support_en.md#tnn-supported-operators)
* [Supported network](./user/support_en.md#tnn-supported-models)
* [Supported devices](./user/support_en.md#tnn-supported-devices)
* [Benchmark & ​​Test method](./user/test_en.md)

## User Manual
* [Overview of usage]()
* [Compile from source](./user/compile_en.md)
* [Toolset]()
    * [Model conversion](./user/convert_en.md)
    * [Model quantification](./user/quantization_en.md)
    * [Model visualization]()
    * [Profiling tool](./development/profiling_en.md)
    * [Function check tool](./development/model_check_en.md)

## API documentation
* [API call](./user/api_en.md)

## Contributing
* [Instructions for basic development](./development/contributing_en.md)
* [Detailed architecture](./development/architecture_en.md)
* [Add OP](./development/add_op_en.md)

## Roadmap
* [Road map]()

## FAQ
* [Frequently asked questions](/doc/en/faq_en.md)