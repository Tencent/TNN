# Demo Introduction

[中文版本](../../cn/user/demo.md)

## I. Introduction to iOS Demo

### How to run

1. Download the Demo model

   ```
   cd <path_to_tnn>/model
   sh download_model.sh
   ```
   PS: If the script cannot download the model due to network problems, please manually create the corresponding folder according to the information in the script and download it by yourself.

2. Open the TNNExamples project

   Enter the directory `<path_to_tnn>/examples/ios/` and double-click to open the TNNExamples project.

3. Set up a developer account

   Click the TNNExamples project as shown below, find the project setting `Signing & Capabilities`, click the Team tab and select `Add an Account...`

  <div align=left><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/ios_add_account_demo.jpg">

   Enter the Apple ID account and password in the following interface. Return to the `Signing & Capabilities` interface, and select the added account in the Team tab. If you don’t have an Apple ID, you can also use the “Create Apple ID” option to apply according to the relevant prompts.

   `PS: There is no fee to apply for Apple ID, it can be passed immediately, and the APP can be run on the real machine after debugging.`

  <div align=left><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/ios_set_account.jpg">

4. Run on real machine

   4.1 Modify `Bundle Identitifier`

   As shown in the figure, after the existing `Bundle Identifier`, a suffix (limited to numbers and letters) is randomly added to avoid personal account conflicts.

  <div align=left><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/ios_set_bundleid_demo.jpg">

  4.2 Verify authorization
     
   For the first time, use the shortcut key `Command + Shift + K` to clean up the project, and then execute the shortcut key` Command + R` to run. If it is the first time to log in with Apple ID, Xcode will pop up a box and report the following error. You need to verify the authorization on the iOS device according to the prompt. Generally speaking, the authorization path on the phone is: Settings-> General-> Profile and Device Management-> Apple Development Options-> Click Trust
     
  <div align=left><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/ios_verify_certificate_demo.jpg">

  4.3 Result
     
   For the first run, use the shortcut key `Command + Shift + K` to clean up the project, and then execute the shortcut key` Command + R` to run. Click the Run button on the interface, the interface will display the CPU and GPU time consumption of all models in the model directory. The running result of the iPhone7 real machine is shown below.
     
  PS:
     
  a) Due to the different GPU and CPU acceleration principles, the GPU performance of a specific model is not necessarily higher than that of the CPU. It is related to the specific model, model structure, and engineering implementation. Everyone is welcome to participate in the development of TNN and make progress together.
     
  b) The macro TNN_SDK_USE_NCNN_MODEL in TNNSDKSample.h defaults to 0, and the TNN model can be set to 1 to run the ncnn model.
     
     c) If you encounter an error message of `Unable to install...`, please delete the existing TNNExamples on the real device and run the installation again.
     
     d) If the CodeSign error `Command CodeSign failed with a nonzero exit code` is encountered when the real device is running, please refer to issue20 `iOS Demo Operation Step Instructions`

### Demo effect 

1. Face detection

   Model source: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

   Effect example: iPhone 7, ARM single thread 6.3206ms

   <div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/face_detector.jpg" width = "50%" height = "50%"/>

2. Image classification

   Model source: https://github.com/forresti/SqueezeNet

   Example: iPhone 7, ARM single thread 13.83ms

   <div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/image_classify.jpg" width = "50%" height = "50%"/>

   
## II. Introduction to Android Demo
### Environment requirements

1. Android Studio 3.5 or above
2. NDK version >= 16

### Steps

1. Download the Demo model

   ```
   cd <path_to_tnn>/model
   sh download_model.sh
   ```

   PS: If the script cannot download the model due to network problems, please manually create the corresponding folder according to the information in the script and download it yourself.
  
   PS for Huawei NPU :
   You need to download the DDK before run the demo. Refer to： [FAQ](../faq_en.md): Huawei NPU Compilation Prerequisite.
 

2. Open the TNNExamples project

   Enter the directory `<path_to_tnn>/examples/android/` and double-click to open the TNN example project.
   
   PS for Huawei NPU ：
   
   1).  After opening the TNN example project，you need to set the TNN_HUAWEI_NPU_ENABLE switch to ON in <path_to_tnn>/examples/android/demo/CMakeList.txt below to use Huawei NPU ：
   
   ````
        set(TNN_HUAWEI_NPU_ENABLE ON CACHE BOOL "" FORCE)
   ````
      
   2). If encountering  `<path_to_tnn>/examples/android/src/main/jni/thirdparty/hiai_ddk/include/graph`Permission Denied，
   Clean Project and rerun.
  
   3). Only Huawei phones of rom version >= 100.320.xxx.xxxx supportS building the example TNN models.
  
   To run the demo, you need to first download the ddk. Refer to ： [FAQ](../faq_en.md) to check the current NPU support and how to update the ROM.

### Running result
1. Face Detection-Pictures
   
   Model source: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

   Effect example: Huawei P30, ARM single thread 32.2359ms

   <div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/android_face_detector_image.jpg" width = "50%" height = " 50%"/>
   
       
   Example： Huawei P30, NPU rom 100.320.010.022 9.04ms
       
   <div align=left ><img src="https://github.com/darrenyao87/tnn-models/blob/master/doc/cn/user/resource/android_face_detecor_image_npu.jpg" width = "50%" height = "50%"/>

2. Face detection-video
   Model source: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

   Effect example: Huawei P30, ARM single thread 122.296ms

   <div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/android_face_detector_stream.jpg" width = "50%" height = " 50%"/>

    Example： Huawei P30, NPU rom 100.320.010.022 28ms
    
    <div align=left ><img src="https://github.com/darrenyao87/tnn-models/blob/master/doc/cn/user/resource/android_face_detector_stream_npu.jpg" width = "50%" height = "50%"/>

3. Image classification

   Model source: https://github.com/forresti/SqueezeNet

   Effect example: Huawei P30, ARM single thread 81.4047ms

   <div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/android_image_classify.jpg" width = "50%" height = " 50%"/>
   
   Example： Huawei P30, NPU rom 100.320.010.022 2.48ms
    
   <div align=left ><img src="https://github.com/darrenyao87/tnn-models/blob/master/doc/cn/user/resource/android_image_classify_npu.jpg" width = "50%" height = "50%"/>
   
## III. Introduction to Linux/Mac/Windows/Armlinux/CudaLinux Demo

### Ability
* Demonstrate the calling method of TNN basic interface, quickly run the model in Linux/Mac/Windows/ArmLinux environment.

### Running Steps
#### 1. Downloand demo models
   ```
   cd <path_to_tnn>/model
   sh download_model.sh
   ```
   If the script cannot download the model due to network problems, please manually create the corresponding folder according to the information in the script and download it yourself.

#### 2. Compile Steps
##### Linux
* Environment Requirements  
   - Cmake (>=3.11)
   - OpenCV3, Can be imported in CMake by find_package(OpenCV 3)
* Compile  
   Move to `examples/x86` directory and execute `build_linux.sh`:
   ```
   cd <path_to_tnn>/examples/x86
   ./build_linux.sh
   ```
* Execute  
   Move to `examples/x86/build_linux` directory and execute:
   ```
   cd build_linux
   
   image-classification demo
   ./demo_x86_imageclassify ../../../model/SqueezeNet/squeezenet_v1.1.tnnproto ../../../model/SqueezeNet/squeezenet_v1.1.tnnmodel

   face-detector demo
   ./demo_x86_facedetector ../../../model/face_detector/version-slim-320_simplified.tnnproto ../../../model/face_detector/version-slim-320_simplified.tnnmodel
   ```

##### MacOS
* Environment Requirements  
   - Cmake (>=3.11)
   - OpenCV3, Can be imported in CMake by ```find_package(OpenCV 3)```. you can install opencv with brew (```brew install opencv@3 && brew link --force opencv@3```).
* Compile  
   Move to `examples/x86` directory and execute `build_macos.sh`:
   ```
   cd <path_to_tnn>/examples/x86
   ./build_macos.sh
   ```
* Execute  
   Move to `examples/x86/build_macos` directory and execute:
   ```
   cd build_macos
   
   image-classification demo
   ./demo_x86_imageclassify ../../../model/SqueezeNet/squeezenet_v1.1.tnnproto ../../../model/SqueezeNet/squeezenet_v1.1.tnnmodel

   face-detector demo
   ./demo_x86_facedetector ../../../model/face_detector/version-slim-320_simplified.tnnproto ../../../model/face_detector/version-slim-320_simplified.tnnmodel

   webcam base face alignment demo
   ./demo_x86_webcam
   ```

##### Windows
* Environment Requirements  
   - Visual Studio (>=2015)
   - Cmake (>=3.11; Or run scripts with Visual Studio Prompt)
   - OpenCV3, compiled by the same version of VC.
* Comiple  
   Open `x64 Native Tools Command Prompt for VS 2017/2019`.
   Move to `examples\x86` directory and execute `build_msvc.bat`:
   ```
   set OpenCV_DIR=`OPENCV_INSTALL_DIR`
   cd <path_to_tnn>\examples\x86
   .\build_msvc.bat 
   ```
* Execute  
   Move to `examples\x86\release` directory and execute：
   ```
   cd release

   image-classification demo
   .\demo_x86_imageclassify ..\..\..\model\SqueezeNet\squeezenet_v1.1.tnnproto ..\..\..\model\SqueezeNet\squeezenet_v1.1.tnnmodel

   face-detector demo
   .\demo_x86_facedetector ..\..\..\model\face_detector\version-slim-320_simplified.tnnproto ..\..\..\model\face_detector\version-slim-320_simplified.tnnmodel

   webcam base face alignment demo
   .\demo_x86_webcam
   ```

##### ArmLinux
* Environment Requirements  
   - Cmake (>=3.1)
   - Install arm toolchain
   - ubuntu:  
      aarch64: sudo apt-get install g++-aarch64-linux-gnu  gcc-aarch64-linux-gnu  
      arm32hf: sudo apt-get install g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf
  - other linux: download toolchains from https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads
* Comiple  
   Move to `examples/linux` directory:
   ```
   cd <path_to_tnn>/examples/linux
   ```
   Modify `build_aarch64.sh` or `build_armhf.sh`，take aarch64 as an example, modify building options:
   ```
   CC=aarch64-linux-gnu-gcc
   CXX=aarch64-linux-gnu-g++
   TNN_LIB_PATH=../../scripts/build_aarch64_linux/
   ```
   execute `build_aarch64.sh`
   ```
   sh build_aarch64.sh
   ```
* Execute  
   Move to `examples/linux/build` directory and execute：
   ```
   cd build

   image-classification demo
   ./demo_arm_linux_imageclassify ../../../model/SqueezeNet/squeezenet_v1.1.tnnproto ../../../model/SqueezeNet/squeezenet_v1.1.tnnmodel

   face-detector demo
   ./demo_arm_linux_facedetector ../../../model/face_detector/version-slim-320_simplified.tnnproto ../../../model/face_detector/version-slim-320_simplified.tnnmodel
   ```

##### CudaLinux
* Environment Requirements
   - Cmake (>= 3.8)
   - CUDA (>= 10.2)
   - TensorRT (>= 7.1)

* Compile
   Configure the TensorRT path in env
   ```
   export TENSORRT_ROOT_DIR=<TensorRT_path>
   ```
   Configure the CuDNN path in env
   ```
   export CUDNN_ROOT_DIR=<CuDNN_path>
   ```
   Move to `example/cuda` directory and execute `build_cuda_linux.sh` :
   ```
   cd <path_to_tnn>/examples/cuda
   sh build_linux.sh
   ```
* Execute
    Move to `examples/cuda/build_cuda_linux` directory and execute:
    ```
    cd build_cuda_linux

    image-classification demo
    ./demo_cuda_imageclassify ../../../model/SqueezeNet/squeezenet_v1.1.tnnproto ../../../model/SqueezeNet/squeezenet_v1.1.tnnmodel

   face-detector demo
   ./demo_cuda_facedetector ~/tnn-models/face-detector/version-slim-320_simplified.tnnproto ~/tnn-models/face-detector/version-slim-320_simplified.tnnmodel
   ```

### Function process
#### Image classification function process
* Create predictor:  
   ```cpp
   auto predictor = std::make_shared<ImageClassifier>();
   ```
* Init predictor:  
   ```cpp
   CHECK_TNN_STATUS(predictor->Init(option));
   // for Linux/Windows
   option->compute_units = TNN_NS::TNNComputeUnitsOpenvino;
   // for ArmLinux
   option->compute_units = TNN_NS::TNNComputeUnitsCPU;
   ```
* Create image_mat:  
   ```cpp
   // for Linux/Windows
   auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_X86, TNN_NS::N8UC3, nchw, data);
   // for ArmLinux
   auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, nchw, data);
   ```
* Run predictor:  
   ```cpp
   CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output));
   ```
#### Face detection function process
* Create predictor:  
   ```cpp
   auto predictor = std::make_shared<UltraFaceDetector>();
   ```
* Init predictor:  
   ```cpp
   CHECK_TNN_STATUS(predictor->Init(option));
   // for Linux/Windows
   option->compute_units = TNN_NS::TNNComputeUnitsOpenvino;
   // for ArmLinux
   option->compute_units = TNN_NS::TNNComputeUnitsCPU;
   ```
* Create image_mat:  
   ```cpp
   // for Linux/Windows
   auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_X86, TNN_NS::N8UC3, nchw, data);
   // for ArmLinux
   auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, nchw, data);
   ```
* Run predictor:  
   ```cpp
   CHECK_TNN_STATUS(predictor->Predict(std::make_shared<UltraFaceDetectorInput>(image_mat), sdk_output));
   ```
* Mark face:  
   ```cpp
   TNN_NS::Rectangle((void *)ifm_buf, image_orig_height, image_orig_width, face.x1, face.y1, face.x2, face.y2, scale_x, scale_y);
   ```


## IV. NCNN model usage and interface introduction

- [NCNN related](./ncnn_en.md)


