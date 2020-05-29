
# Demo Introduction

## I. Introduction to iOS Demo

### How to run

1. Download the Demo model

   ```
   cd <path_to_tnn>/model
   sh download_model.sh
   ```

   PS: If the script cannot download the model due to network problems, please manually create the corresponding folder according to the information in the script and download

2. Run the TNNExamples project

   Enter the directory `<path_to_tnn>/examples/ios /` and open the TNNExamples project. The default UI is for face detection, you can click the edit button in the upper right corner to switch between different functions such as image classification.

   PS: The macro TNN_SDK_USE_NCNN_MODEL in TNNSDKSample.h is set to 0 by default to run TNN model. It can be set to 1 to run the ncnn model.

3. Modify the signature and run on the real machine

   Modify the signature in benchmark project and tnn dependent project setting `Signing & Capabilities` according to your Apple developer account. If you encounter signature conflicts about your personal account, you can try to modify` Bundle Identifier`

   
### Demo effect 

1. Face detection

   Model source: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

   Effect example: iPhone 7, ARM single thread 6.3206ms

   <div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/user/resource/face_detector.jpg" width = "33%" height = "33%"/>

2. Image classification

   Model source: https://github.com/forresti/SqueezeNet

   Example: iPhone 7, ARM single thread 13.83ms

   <div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/user/resource/image_classify.jpg" width = "33%" height = "33%"/>

   
## II. Introduction to Android Demo

TODO

## III. Introduction to Armlinux Demo

### Demo the usage of TNN lib and run your model quickly on linux platform
* How to run: 
    * Look at [arm linux demo Readme](../../../examples/armlinux/Readme.md)

### Init function flow

1. Specify the model file path in TNN_NS::ModelConfig and create a TNN_NS::TNN instance.

Related code:

    TNN_NS :: ModelConfig model_config;
    model_config.params.push_back (buffer);
    model_config.params.push_back (model_file);
    CHECK_TNN_STATUS (tnn_.Init (model_config));

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

2. Forward calculation.

Related code:

    CHECK_TNN_STATUS (net_instance _-> Forward ());

3. Data transferring and post-processing.

Related code:

    TNN_NS :: BlobConverter output_blob_convert (output_blobs_.begin ()-> second);
    CHECK_TNN_STATUS (
        output_blob_convert.ConvertToMat (output_mat, output_convert_param_, nullptr));


## IV. NCNN model usage and interface introduction

- [NCNN related](./ncnn_en.md)