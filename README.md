[中文版本](README_CH.md)
<div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/TNN.png"/>

## Introduction

TNN: A high-performance, lightweight neural network inference framework open sourced by Tencent Youtu Lab. It also has many outstanding advantages such as cross-platform, high performance, model compression, and code tailoring. The TNN framework further strengthens the support and performance optimization of mobile devices on the basis of the original Rapidnet and ncnn frameworks. At the same time, it refers to the high performance and good scalability characteristics of the industry's mainstream open source frameworks, and expands the support for X86 and NV GPUs. On the mobile phone, TNN has been used by many applications such as mobile QQ, weishi, and Pitu. As a basic acceleration framework for Tencent Cloud AI, TNN has provided acceleration support for the implementation of many businesses. Everyone is welcome to participate in the collaborative construction to promote the further improvement of the TNN reasoning framework.

## Effect Example

Face Detection(blazeface)   |   Object Detection(yolov5s)    |  Face Alignment<br>(from Tencent Youtu Lab)  |   Hair Segmentation<br>(from Tencent Guangying Lab) 
:-------------------------: | :------: | :------: | :------:
[![face_detection](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/face_detection.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/blazeface) <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/blazeface) | [![yolov5](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/object-detection.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/yolov5) <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/yolov5) | [![youtu_face_alignment](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/face_alignment.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/youtu_face_alignment) <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/youtu_face_alignment) | [![hair_segmentation](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/hair_seg_red.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/hair_segmentation) <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/hair_segmentation)

Pose Estimation<br>(from Tencent Guangliu)   |   Pose Estimation<br>(blazepose)    |   Chinese OCR |  Reading Comprehension
:--------------------------: | :------: | :------: | :------:
[![skeleton](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/skeleton_guangliu.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/skeleton) <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/skeleton) | [![blazepose](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/skeleton_blazepose.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/blazepose) <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/blazepose) | [![chinese-ocr](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/chinese-ocr.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/chinese-ocr) <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/chinese-ocr) | [![bertsquad10](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/bert_squad.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/bertsquad10) <br> [model link](https://github.com/darrenyao87/tnn-models/tree/master/model/bertsquad10)

The support for each demo is shown in the following table. You can click the ✅ and find the entrance code for each demo.
demo                                                                                      |   ARM    |  OpenCL  |   Metal  |    NPU   |
:---------------------------------------------------------------------------------------: | :------: | :------: | :------: | :------: |
[Face Detection](https://github.com/Tencent/TNN/blob/master/examples/base/blazeface_detector.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamBlazeFaceDetector) | [✅ ](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNBlazeFaceDetectorViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamBlazeFaceDetector) |
[Object Detection](https://github.com/Tencent/TNN/blob/master/examples/base/object_detector_yolo.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamObjectDetector) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNYoloObjectDetectorViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamObjectDetector) |
[Face Alignment](https://github.com/Tencent/TNN/blob/master/examples/base/face_detect_aligner.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamBlazeFaceAlign) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNFaceDetectAlignerViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamBlazeFaceAlign) |
[Hair Segmentation](https://github.com/Tencent/TNN/blob/master/examples/base/hair_segmentation.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamHairSegmentation) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNHairSegmentationViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamHairSegmentation) |
[Pose Estimation<br>(from Tencent Guangliu)](https://github.com/Tencent/TNN/blob/master/examples/base/skeleton_detector.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamSkeletonDetector) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNSkeletonDetectorViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamSkeletonDetector) |
[Pose Estimation(blazepose)](https://github.com/Tencent/TNN/blob/master/examples/base/pose_detect_landmark.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamPoseDetectLandmark) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNPoseDetectLandmarkViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamPoseDetectLandmark) |
[Chinese OCR](https://github.com/Tencent/TNN/blob/master/examples/base/ocr_driver.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamOCRDetector) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNOCRViewModel.mm) |  |

## Quick Start

It is very simple to use TNN. If you have a trained model, the model can be deployed on the target platform through three steps.
1. Convert the trained model into a TNN model. We provide a wealth of tools to help you complete this step, whether you are using Tensorflow, Pytorch, or Caffe, you can easily complete the conversion.
Detailed hands-on tutorials can be found here [How to Create a TNN Model](doc/en/user/convert_en.md).

2. When you have finished converting the model, the second step is to compile the TNN engine of the target platform. You can choose among different acceleration solutions such as ARM/OpenCL/Metal/NPU/X86/CUDA according to the hardware support.
   For these platforms, TNN provides convenient one-click scripts to compile. For detailed steps, please refer to [How to Compile TNN](doc/en/user/compile_en.md).

3. The final step is to use the compiled TNN engine for inference. You can make program calls to TNN inside your application. We provide a rich and detailed demo as a reference to help you complete.
    * [Run an iOS Demo](doc/en/user/demo_en.md)
    * [Run an Android Demo](doc/en/user/demo_en.md)
    * [Run an Linux/Windows Demo](doc/en/user/demo_en.md)

## Technical Solutions

At present, TNN has been launched in various major businesses, and its following characteristics have been widely praised.

* Computation optimization
    * The backend operators are primely optimized to make the best use of computing power in different architectures, regarding instruction issue, throughput, delay, cache bandwidth, cache delay, registers, etc..
    * The TNN performance on mainstream hardware platforms (CPU: ARMv7, ARMv8， X86, GPU: Mali, Adreno, Apple， NV GPU， NPU) has been greatly tuned and improved.
    * The convolution function is implemented by various algorithms such as Winograd, Tile-GEMM, Direct Conv, etc., to ensure efficiency under different parameters and sizes.
    * Op fusion: TNN can do offline analysis of network graph, fuse multiple simple operations and reduce overhead such as redundant memory access and kernel startup cost.

* Low precision computation acceleration
    * TNN supports INT8/FP16 mode, reduces model size & memory consumption, and utilizes specific hardware low-precision instructions to accelerate calculations.
    * TNN supports INT8 WINOGRAD algorithm, (input 6bit), further reduces the model calculation complexity without sacrificing the accuracy.
    * TNN supports mixed-precision data in one model, speeding up the model's calculation speed while preserving its accuracy.

* Memory optimization
    * Efficient "memory pool" implementation: Based on a full network DAG analysis, the implementation reuses memory between non-dependent nodes which reduces memory cost by 90%.
    * Cross-model memory reduces: This supports external real-time design for network memory so that multiple models can share mutual memory.

* The performance of mainstream models on TNN: [benchmark data](benchmark_data.md)

* TNN architecture diagram：

   <div><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/en/imgs/tnn_architect.jpg"/>

* TNN supports TensorFlow, Pytorch, MxNet, Caffe, and other training frameworks through ONNX, leveraging the continuous improvement of the ONNX open-source society.
  Currently, TNN supports 100+ ONNX operators, consisting of most of the mainstream CNN, NLP operators needed.
* TNN runs on mainstream operating systems (Android, iOS, embedded Linux, Windows, Linux), and is compatible with ARM CPU,X86 GPU, NPU hardware platform.
* TNN is constructed through Modular Design, which abstracts and isolates components such as model analysis, graph construction, graph optimization, low-level hardware adaptation, and high-performance kernel.
   It uses "Factory Mode" to register and build devices, that tries to minimize the cost of supporting more hardware and acceleration solutions.
* The size of the mobile dynamic library is only around 400KB, and it provides basic image conversion operations, which are light-weight and convenient. TNN uses unified models and interfaces across platforms and can switch easily by configuring just one single parameter.

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

* [onnxruntime](https://github.com/microsoft/onnxruntime)

* [openvino](https://github.com/openvinotoolkit/openvino)

## License
* [BSD 3 Clause](LICENSE)

## FAQ
* [FAQ](doc/en/faq_en.md)

## Join Us

* Everyone is welcome to participate to build the best inference framework in the industry.

* Technical Discussion QQ Group: 913940506 Answer: TNN

* Scan the QR code to join the TNN discussion group：
<div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/TNN-QQ.png"/>
