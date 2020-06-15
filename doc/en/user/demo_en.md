# Demo Introduction

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

2. Open the TNNExamples project

   Enter the directory `<path_to_tnn>/examples/android/` and double-click to open the TNNExamples project.

### Running result
1. Face Detection-Pictures
   
   Model source: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

   Effect example: Huawei P30, ARM single thread 32.2359ms

   <div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/android_face_detector_image.jpg" width = "50%" height = " 50%"/>

2. Face detection-video
   Model source: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

   Effect example: Huawei P30, ARM single thread 122.296ms

   <div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/android_face_detector_stream.jpg" width = "50%" height = " 50%"/>

3. Image classification

   Model source: https://github.com/forresti/SqueezeNet

   Effect example: Huawei P30, ARM single thread 81.4047ms

   <div align=left ><img src="https://gitee.com/darren3d/tnn-resource/raw/master/doc/cn/user/resource/android_image_classify.jpg" width = "50%" height = " 50%"/>

## III. Introduction to Armlinux Demo

### Ability
* Demonstrate the calling method of TNN basic interface, quickly run the model in Linux environment.

### Compile
* Refer to[arm linux Readme文档](/examples/armlinux/Readme.md)

### Init function flow

1. Specify the model file path in TNN_NS::ModelConfig and create a TNN_NS::TNN instance.

Related code:

    TNN_NS::ModelConfig model_config;
    model_config.params.push_back(buffer);
    model_config.params.push_back(model_file);
    CHECK_TNN_STATUS(tnn_.Init(model_config));

2. Specify the device type and other information in TNN_NS::NetworkConfig, then create a TNN_NS::Instance instance.

Related code:

    TNN_NS :: NetworkConfig config;
    config.device_type = TNN_NS :: DEVICE_ARM;
    TNN_NS :: Status error;
    net_instance_ = tnn_.CreateInst (config, error);
    CHECK_TNN_STATUS (error);

3. Obtain input and output information.

Related code:

    CHECK_TNN_STATUS (net_instance _-> GetAllInputBlobs (input_blobs_));
    CHECK_TNN_STATUS (net_instance _-> GetAllOutputBlobs (output_blobs_));

### Forward function flow

1. Preprocessing and data transfer.

Related code:

    TNN_NS :: BlobConverter input_blob_convert (input_blobs_.begin ()-> second);
    CHECK_TNN_STATUS (
        input_blob_convert.ConvertFromMat (input_mat, input_convert_param_, nullptr));

2. Forward calculation.w

Related code:

    CHECK_TNN_STATUS (net_instance _-> Forward ());

3. Data transferring and post-processing.

Related code:

    TNN_NS :: BlobConverter output_blob_convert (output_blobs_.begin ()-> second);
    CHECK_TNN_STATUS (
        output_blob_convert.ConvertToMat (output_mat, output_convert_param_, nullptr));


## IV. NCNN model usage and interface introduction

- [NCNN related](./ncnn_en.md)
