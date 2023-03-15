[English Version](README.md)
<div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/TNN.png"/>

## 简介

TNN：由腾讯优图实验室开源的高性能、轻量级神经网络推理框架，同时拥有跨平台、高性能、模型压缩、代码裁剪等众多突出优势。TNN框架在原有Rapidnet、ncnn框架的基础上进一步加强了移动端设备的支持以及性能优化，同时借鉴了业界主流开源框架高性能和良好拓展性的特性，拓展了对于后台X86, NV GPU的支持。手机端 TNN已经在手机QQ、微视、P图等众多应用中落地，服务端TNN作为腾讯云AI基础加速框架已为众多业务落地提供加速支持。欢迎大家参与协同共建，促进TNN推理框架进一步完善。


## 效果示例

人脸检测(blazeface)   |  人脸配准(腾讯优图)  |   头发分割(腾讯光影)
:-------------------------: | :------: | :------:
[![face_detection](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/face_detection.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/blazeface) <br> 模型链接: [tflite](https://github.com/google/mediapipe/blob/master/mediapipe/models/face_detection_front.tflite) [tnn](https://github.com/darrenyao87/tnn-models/tree/master/model/blazeface) | [![youtu_face_alignment](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/face_alignment.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/youtu_face_alignment) <br> 模型链接: [tnn](https://github.com/darrenyao87/tnn-models/tree/master/model/youtu_face_alignment) | [![hair_segmentation](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/hair_seg_red.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/hair_segmentation) <br> 模型链接: [tnn](https://github.com/darrenyao87/tnn-models/tree/master/model/hair_segmentation)

姿势估计(腾讯光流)   |   姿势估计(blazepose)    |  中文字符识别 
:--------------------------: | :------: | :------: 
[![skeleton](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/skeleton_guangliu.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/skeleton) <br> 模型链接: [tnn](https://github.com/darrenyao87/tnn-models/tree/master/model/skeleton) | [![blazepose](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/skeleton_blazepose.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/blazepose) <br> 模型链接: [tflite](https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_full.tflite) [tnn](https://github.com/darrenyao87/tnn-models/tree/master/model/blazepose) | [![chinese-ocr](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/chinese-ocr.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/chinese-ocr) <br> 模型链接: [onnx](https://github.com/DayBreak-u/chineseocr_lite/tree/onnx/models) [tnn](https://github.com/darrenyao87/tnn-models/tree/master/model/chinese-ocr)

物体检测(yolov5s)    |    物体检测(MobilenetV2-SSD)    |  阅读理解
:-------------------------: |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:| :------: 
[![yolov5](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/object-detection.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/yolov5) <br> 模型链接: [onnx](https://github.com/ultralytics/yolov5/blob/master/export.py) [tnn](https://github.com/darrenyao87/tnn-models/tree/master/model/yolov5) | [![mobilenetv2_ssd](tutorial/mobilenet_v2_ssd/imgs/mobilenetv2_ssd_tf_fix_box.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/mobilenet_v2-ssd) <br> 模型链接: [tensorflow](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz) [tnn](https://github.com/darrenyao87/tnn-models/tree/master/model/mobilenet_v2-ssd) |  [![bertsquad10](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/demo/bert_squad.gif)](https://github.com/darrenyao87/tnn-models/tree/master/model/bertsquad10) <br> 模型链接: [onnx](https://github.com/onnx/models/blob/master/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx) [tnn](https://github.com/darrenyao87/tnn-models/tree/master/model/bertsquad10)

<small>中文字符识别demo是[chineseocr_lite](https://github.com/DayBreak-u/chineseocr_lite)的TNN实现，是一个超轻量级的中文ocr，支持倾斜、旋转和竖排文字识别。</small>

各个平台对demo的支持情况如下表所示，单击✅标记，便可以跳转至对应demo的入口代码。

demo                                                                                      |   ARM    |  OpenCL  |   Metal  | Huawei NPU | Apple NPU |    X86   |    CUDA   
:---------------------------------------------------------------------------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
[人脸检测](https://github.com/Tencent/TNN/blob/master/examples/base/blazeface_detector.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamBlazeFaceDetector) | [✅ ](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNBlazeFaceDetectorViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamBlazeFaceDetector) | [✅ ](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNBlazeFaceDetectorViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/linux/src/TNNFaceDetector/TNNFaceDetector.cc) | [✅](https://github.com/Tencent/TNN/tree/master/examples/linux/src/TNNFaceDetector/TNNFaceDetector.cc)
[物体检测](https://github.com/Tencent/TNN/blob/master/examples/base/object_detector_yolo.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamObjectDetector) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNYoloObjectDetectorViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamObjectDetector) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNYoloObjectDetectorViewModel.mm) | [✅](https://github.com/Tencent/TNN/blob/master/examples/linux/src/TNNObjectDetector/TNNObjectDetector.cc) | [✅](https://github.com/Tencent/TNN/blob/master/examples/linux/src/TNNObjectDetector/TNNObjectDetector.cc)
[人脸配准](https://github.com/Tencent/TNN/blob/master/examples/base/face_detect_aligner.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamBlazeFaceAlign) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNFaceDetectAlignerViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamBlazeFaceAlign) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNFaceDetectAlignerViewModel.mm) | [✅](https://github.com/Tencent/TNN/blob/master/examples/linux/src/TNNFaceAligner/TNNFaceAligner.cc) | [✅](https://github.com/Tencent/TNN/blob/master/examples/linux/src/TNNFaceAligner/TNNFaceAligner.cc)
[头发分割](https://github.com/Tencent/TNN/blob/master/examples/base/hair_segmentation.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamHairSegmentation) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNHairSegmentationViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamHairSegmentation) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNHairSegmentationViewModel.mm)| [✅](https://github.com/Tencent/TNN/blob/master/examples/linux/src/TNNHairSegmentation/TNNHairSegmentation.cc) | [✅](https://github.com/Tencent/TNN/blob/master/examples/linux/src/TNNHairSegmentation/TNNHairSegmentation.cc) 
[姿势估计(腾讯光流)](https://github.com/Tencent/TNN/blob/master/examples/base/skeleton_detector.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamSkeletonDetector) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNSkeletonDetectorViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamSkeletonDetector) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNSkeletonDetectorViewModel.mm) | [✅](https://github.com/Tencent/TNN/blob/master/examples/linux/src/TNNSkeletonDetector/TNNSkeletonDetector.cc) | [✅](https://github.com/Tencent/TNN/blob/master/examples/linux/src/TNNSkeletonDetector/TNNSkeletonDetector.cc)
[姿势估计(blazepose)](https://github.com/Tencent/TNN/blob/master/examples/base/pose_detect_landmark.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamPoseDetectLandmark) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNPoseDetectLandmarkViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamPoseDetectLandmark) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNPoseDetectLandmarkViewModel.mm) | [✅](https://github.com/Tencent/TNN/tree/master/examples/linux/src/TNNBlazePose/TNNBlazePose.cc) | [✅](https://github.com/Tencent/TNN/tree/master/examples/linux/src/TNNBlazePose/TNNBlazePose.cc) 
[中文字符识别](https://github.com/Tencent/TNN/blob/master/examples/base/ocr_driver.cc)   | ✅ | [✅](https://github.com/Tencent/TNN/tree/master/examples/android/demo/src/main/java/com/tencent/tnn/demo/StreamOCRDetector) | [✅](https://github.com/Tencent/TNN/blob/master/examples/ios/TNNExamples/TNNCameraPreviewController/TNNViewModel/TNNOCRViewModel.mm) | | | [✅](https://github.com/Tencent/TNN/tree/master/examples/linux/src/TNNOcrDetector/TNNOcrDetector.cc) | [✅](https://github.com/Tencent/TNN/tree/master/examples/linux/src/TNNOcrDetector/TNNOcrDetector.cc)
[阅读理解](https://github.com/Tencent/TNN/blob/master/examples/base/bert_tokenizer.cc)  | | | | | | [✅](https://github.com/Tencent/TNN/blob/master/examples/linux/src/BertReadingComprehension/BertReadingComprehension.cc) | [✅](https://github.com/Tencent/TNN/blob/master/examples/linux/src/BertReadingComprehension/BertReadingComprehension.cc)

## 快速开始

使用 TNN 非常简单，如果你有一个已经训练好的模型, 那么一般而言通过以下三个步骤就能完成模型在目标平台上的部署。
1. 第一步是把训练好的模型转换成TNN的模型，为此我们提供了丰富的工具来帮助你完成这一步，无论你使用的是 TensorFlow、PyTorch、或者 Caffe，都可以轻松完成转换。
详细的手把手教程可以参见这里[如何转换模型](doc/cn/user/convert.md)。

2. 当你完成了模型的转换，第二步就是编译目标平台的 TNN 引擎了，你可以根据自己的目标平台的硬件支持情况，选择 CPU/ARM/OpenCL/Metal/NPU/X86/CUDA 等加速方案。
   对于这些平台，TNN 都提供了一键编译的脚本，使用非常方便。详细步骤可以参考这里[如何编译TNN](doc/cn/user/compile.md)。

3. 最后一步就是使用编译好的 TNN 引擎进行推理，你可以在自己的应用程序中嵌入对 TNN 的调用，这方面我们提供了丰富而详实的 demo 来帮助你完成。
    *  [从0开始跑通一个iOS Demo](doc/cn/user/demo.md)
    *  [从0开始跑通一个Android Demo](doc/cn/user/demo.md)
    *  [从0开始跑通一个Windows/Linux Demo](doc/cn/user/demo.md)

## 技术方案

目前TNN具有的以下特性获得了广泛的好评。

* 计算优化
    * 针对不同架构在硬件指令发射、吞吐、延迟、缓存带宽、缓存延迟、寄存器数量等特点，深度优化底层算子，极致利用硬件算力
    * 主流硬件平台(CPU: ARMv7， ARMv8，X86 GPU: Mali， Adreno， Apple， NV GPU) 深度调优
    * CNN 核心卷积运算通过 Winograd，Tile-GEMM， Direct Conv 等多种算法实现，保证不同参数、计算尺度下高效计算
    * Op 融合：离线分析网络计算图，多个小 Op（计算量小、功能较简单）融合运算，减少反复内存读取、kernel 启动等开销

* 低精度优化
    * 支持 INT8， FP16 低精度计算，减少模型大小、内存消耗，同时利用硬件低精度计算指令加速计算
    * 支持 INT8 Winograd 算法，(输入6bit)， 在精度满足要求的情况下，进一步降低模型计算复杂度
    * 支持单模型多种精度混合计算，加速计算同时保证模型精度

* 内存优化
    * 高效”内存池”实现：通过 DAG 网络计算图分析，实现无计算依赖的节点间复用内存，降低 90% 内存资源消耗
    * 跨模型内存复用：支持外部实时指定用于网络内存，实现“多个模型，单份内存”。

* 主流模型实测性能：[评测数据](doc/benchmark_data.md)

* TNN架构图：

   <div><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/imgs/tnn_architect.jpg"/>

* 通过 ONNX 支持 TensorFlow， PyTorch， MXNet， Caffe 等多种训练框架，充分利用和融入不断完善的 ONNX 开源生态。当前支持 ONNX 算子100+，覆盖主流CNN, NLP网络。
* 支持主流安卓、iOS、Embedded Linux 操作系统, Windows, Linux，支持 ARM CPU, x86, Mali GPU, Adreno GPU, NV GPU, 达芬奇NPU，RK NPU。
* 模块化设计，将模型解析、计算图构建、优化、底层硬件适配、高性能 kernel 实现各部分抽象隔离，通过 Factory Mode 注册、构建设备，方便接入更多的底层硬件、加速方案。
* 移动端动态库尺寸仅约 400KB，并提供基础图像变换操作，调用简单便捷。跨平台模型统一、调用接口统一，通过单个配置参数快速切换。

## 能力展示
* [支持的算子](doc/cn/user/support.md)
* [支持的网络](doc/cn/user/support.md)
* [支持的架构](doc/cn/user/support.md)
* [Benchmark性能测试方法](doc/cn/development/profiling.md)

## 使用手册
* [从源码编译](doc/cn/user/compile.md)
* [工具集]()
    * [模型转换](doc/cn/user/convert.md)
    * [模型量化](doc/cn/user/quantization.md)
    * [模型可视化Netron](https://lutzroeder.github.io/netron/)
    * [性能分析工具](doc/cn/development/profiling.md)
    * [模型对齐工具](doc/cn/development/model_check.md)
* [教程]()
    * [基于 SSD 的 TNN 模型转换和部署](tutorial/mobilenet_v2_ssd/doc/ssd_conversion_and_deployment.md)

## API文档
* [API调用](doc/cn/user/api.md)

## 贡献者须知
* [开发基础须知](doc/cn/development/contributing.md)
* [架构详解](doc/cn/development/architecture.md)
* [新增OP](doc/cn/development/add_op.md)
* [单元测试](doc/cn/development/unit_test.md)

## Roadmap
* [Road map](doc/cn/user/roadmap.md)

## 致谢
TNN参考和借鉴了下列项目：

* [ncnn](https://github.com/Tencent/ncnn)
* [mace](https://github.com/XiaoMi/mace.git)
* [MNN](https://github.com/alibaba/MNN)
* [caffe-onnx](https://github.com/htshinichi/caffe-onnx)
* [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)
* [onnx](https://github.com/onnx/onnx)
* [onnxruntime](https://github.com/microsoft/onnxruntime)
* [openvino](https://github.com/openvinotoolkit/openvino) 
* [xbyak](https://github.com/herumi/xbyak)
* [TensorRT](https://developer.nvidia.com/zh-cn/tensorrt)

## License

* [BSD 3 Clause](LICENSE)

## FAQ
* [FAQ 常见问题](doc/cn/faq.md)

## 加入我们

* 欢迎大家参与，协同共建，打造业界最好的高性能推理框架。

* 技术交流 QQ 群： 704900079 答案：TNN

* QQ 群二维码：
<div align=left ><img src="TNN-QQ.png"/>
