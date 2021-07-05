# API Documentation

[中文版本](../../cn/user/api.md)

## I. API Interface Compatibility

All exposed interfaces of TNN are displayed and declared by PUBLIC macro, while non-exposed interfaces and symbols are not visible to external.

```cpp
#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define PUBLIC __attribute__ ((dllexport))
    #else
      #define PUBLIC __declspec (dllexport)
    #endif
  #else
    #ifdef __GNUC__
      #define PUBLIC __attribute__ ((dllimport))
    #else
      #define PUBLIC __declspec (dllimport)
    #endif
  #endif
  #define LOCAL
#else
  #if __GNUC__> = 4
    #define PUBLIC __attribute__ ((visibility ("default")))
    #define LOCAL __attribute__ ((visibility ("hidden")))
  #else
    #define PUBLIC
    #define LOCAL
  #endif
#endif
```

Compatibility of different API versions follows[Semantic Version 2.0.0](https://semver.org) rules.

## II. API Call

### Introduction
The API call mainly introduces the four steps: model analysis, network construction, input setting, and output acquisition. For detailed description, please refer to the API detailed explanation section.

### Step1. Model analysis

```cpp
TNN tnn;
TNN_NS::ModelConfig model_config;
//proto file content saved to proto_buffer
model_config.params.push_back(proto_buffer);
//model file content saved to model_buffer
model_config.params.push_back(model_buffer);
tnn.Init(model_config);
```

TNN model analysis needs to configure the ModelConfig parameter, pass in the content of proto and model files, and call the TNN Init interface to complete the model analysis.


### Step2. Network construction

```cpp
TNN_NS::NetworkConfig config;
config.device_type = TNN_NS::DEVICE_ARM;
TNN_NS::Status error;
auto net_instance = tnn.CreateInst(config, error);
```

TNN network construction needs configure the NetworkConfig parameter，and device_type could be set as `DEVICE_ARM`， `DEVICE_OPENCL`， `DEVICE_METAL`， `DEVICE_X86`， `DEVICE_CUDA`, `DEVICE_HUAWEI_NPU`, `DEVICE_RK_NPU` or other acceleration method，the construction of the network is completed through CreateInst interface.

```cpp
config.network_type = TNN_NS::NETWORK_TYPE_HUAWEI_NPU;
```

### Step3. Input

```cpp
    auto status = instance->SetInputMat(input_mat, input_cvt_param);
```

TNN input is set by SetInputMat interface.The data to be passed in is saved to input_mat.

### Step 4. Network Infer

```cpp
auto status = instante->Forward();
```
The TNN Forward interface is a synchronous call interface, and the ForwardAsync interface is an asynchronous call interface.

### Step5. Output 

```cpp
    auto status = instance->GetOutputMat(output_mat);
```

TNN output is exported by ObtainingGetOutputMat interface. The result would be saved to output_mat in a specific format.

## III. API Explanation

### API directory structure 

```bash
.
└── tnn
    ├── core
    │   ├── blob.h                  # data transfer
    │   ├── common.h                # define common structure
    │   ├── instance.h              # netwrok instance
    │   ├── macro.h                 # common macro definition
    │   ├── mat.h                   # input interface，like cv::Mat
    │   ├── status.h                # interface status
    │   └── tnn.h                   # model analysis
    ├── utils
    │   ├── bfp16_utils.h           # bfp16 conversion tool
    │   ├── blob_converter.h        # blob input/output tool
    │   ├── cpu_utils.h             # CPU performance specific optimization tool
    │   ├── data_type_utils.h       # data type conversion tool
    │   ├── dims_vector_utils.h     # dim size calculation tool
    │   ├── half_utils.h            # fp16 conversion tool
    │   ├── mat_utils.h             # mat conversion tool
    │   └── string_utils.h          # string conversion tool
    └── version.h                   # Compile and build information
```

### 1. core/common.h
`DataType`: Define enumeration values ​​for different datatypes.  
`DataFormat`: Define the different data arrangement methods of Blob Data.  
`NetworkType`: define different network construction types, build TNN network by default, also support third-party library network construction.  
`DeviceType`: Used to specify the device the network running on and the corresponding acceleration method.  
`ModelType`: define the model type, default is TNN model, also supports the import of other third-party library model formats.  
`Precision`: Define the accuracy of the network operation.  

```cpp
struct PUBLIC ModelConfig {

    ModelType model_type = MODEL_TYPE_TNN;

    // tnn model need two params: order is proto content, model content.
    // ncnn need two: params: order is param content, bin content.
    // openvino model need two params: order is xml content, model path.
    // coreml model need one param: coreml model directory path.
    // snpe model need one param: dlc model directory path.
    // hiai model need two params: order is model name, model file path.
    // atlas model need one param: config string.
    std::vector<std::string> params;
};
```

ModelConfig parameters：  

- `model_type`: The current open source version of TNN only supports importing `MODEL_TYPE_TNN`, `MODEL_TYPE_NCNN`, `MODEL_TYPE_COREML` model formats.  
- `params`: The TNN model needs to pass in the content of the proto file and the path of the model file. The NCNN model needs to input the content of the param file and the path of the bin file, and the COREML model needs to input the directory path where the coreml model is located.  

```cpp
struct PUBLIC NetworkConfig {
    // device type default cpu
    DeviceType device_type = DEVICE_ARM;

    // device id default 0
    int device_id = 0;

    // blob data format, auto decided by device
    DataFormat data_format = DATA_FORMAT_AUTO;

    // network type, auto decided by device
    NetworkType network_type = NETWORK_TYPE_AUTO;

    // raidnet instances not share memory with others
    ShareMemoryMode share_memory_mode = SHARE_MEMORY_MODE_DEFAULT;

    // dependent library path
    std::vector<std::string> library_path = {};

    // compute precision
    Precision precision = PRECISION_AUTO;

    // cache path to store possible cache models or opt kernel or opencl program cache
    std::string cache_path = "";

    // network init or reshape may cost more time to select opt kernel implement if enable tune kernel
    // cache_path can set to store tune kernel info.
    bool enable_tune_kernel = false;
};
```
NetworkConfig parameter description:  

- `device_type`:   
The default is `DEVICE_ARM`. `DEVICE_NAIVE`, `DEVICE_ARM`, `DEVICE_X86`, `DEVICE_OPENCL`, `DEVICE_METAL`, `DEVICE_CUDA`, `DEVICE_HUAWEI_NPU`, `DEVICE_RK_NPU` are currently supported.  
- `device_id`: The default value is 0. Multiple devices can be selected by `device_id`. Currently, only `DEVICE_CUDA` needs to configure this parameter to specify the gpu id.  
- `data_format`: By default, tnn automatically selects the blob data arrangement method for acceleration. You can set a specific blob data arrangement for acceleration through this parameter.  
- `network_type`: By default, the network type is automatically selected according to the `device_type`, and the network type to be constructed can be specified.  
- `share_memory_mode`: tnn instance memory sharing mode.  
- `library_path`: support external dependent library loading, this parameter needs to be configured when the iOS metal kernel library is placed in the app non-default path.  
- `precision`: Network precision type. The precision is automatically selected according to different `device_type` by default.  
- `cache_path`: Huawei NPU specifies the cache path to store the om files transferred during operation, and subsequent operations can directly load the corresponding om files through the cache path. OpenCL specifies the cache path to store the compiled binary files of kernel, and subsequent initialization can directly create kernals through the binary cache files. If `enable_tune_kernel` is turned on, you can store the tune parameters by specifying the cache path, and then you can load the tune parameters directly without having to tune the kernel every time you run it.

```cpp
typedef enum {
    // default
    SHARE_MEMORY_MODE_DEFAULT = 0,
    // same thread tnn instance share blob memory
    SHARE_MEMORY_MODE_SHARE_ONE_THREAD = 1,
    // set blob memory from external, different thread share blob memory need
    // synchronize
    SHARE_MEMORY_MODE_SET_FROM_EXTERNAL = 2
} ShareMemoryMode;
```

- `SHARED_MEMORY_MODE_DEFAULT`: only supports memory sharing between different blobs of the same instance.  
- `SHARE_MEMORY_MODE_SHARE_ONE_THREAD`: supports memory sharing of different instances of the same thread.  
- `SHARE_MEMORY_MODE_SET_FROM_EXTERNAL`: supports instance memory to be passed in from outside, the sharing mode is determined by the calling side, synchronization among threads needs to deal with synchronization issues, and memory allocation and release all require maintenance on the calling side.  

### 2. core/tnn.h

```cpp
class PUBLIC TNN {
public:
    ...

    // init tnn implement, interpret model.
    Status Init(ModelConfig& config);

    // denit tnn implement, release model interpreter.
    Status DeInit();

    // add output to the model.
    // if output_name of blob not found, then search output_index of layer.
    Status AddOutput(const std::string& output_name, int output_index = 0);

    // return input shapes map from model
    Status GetModelInputShapesMap(InputShapesMap& shapes_map);

    // create tnn network instance with network config and inputs shape.
    // if inputs shape not set, use default from model.
    std::shared_ptr<Instance> CreateInst(
        NetworkConfig& config, Status& status,
        InputShapesMap inputs_shape = InputShapesMap());

    // create tnn network instance with network config and min max inputs shape,
    // instance reshape can support range from min inputs shape to max inputs shape.
    std::shared_ptr<Instance> CreateInst(
        NetworkConfig& config, Status& status,
        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape);

    ...
};
```

TNN interface description:  

- Init interface: responsible for importing and parsing model data, need to configure and import ModelConfig.  
- DeInit interface: responsible for the release of tnn implement, the default destructor can be automatically released.  
- AddOutput interface: support to increase the model output, you can define any layer of network output as the model output.  
- CreateInst interface: responsible for network instance Instance construction.  
- GetModelInputShapesMap interface: Get the model input size parsed by the model.  
- CreateInst interface: responsible for the construction of the network instance. If the input dimensions are variable during operation, you need to configure `min_inputs_shape` and `max_inputs_shape` to specify the maximum and minimum dimensions supported by each dimension of the input.  

### 3. core/instance.h

```cpp
class PUBLIC Instance {
public:
    Instance(NetworkConfig& net_config, ModelConfig& model_config);

    ~Instance();

    // init with model interpeter and inputs shape.
    Status Init(std::shared_ptr<AbstractModelInterpreter> interpreter, InputShapesMap inputs_shape);

    // deinit, release network
    Status DeInit();

    //  return memory bytes required for forward
    Status GetForwardMemorySize(int& memory_size);

    //  set memory to tnn instance. if success, return status code zero.
    //  only instance created with SHARE_MEMORY_MODE_SET_FROM_EXTERNAL can be set from external.
    //  the memory size need >=  GetForwardMemorySize().
    //  releasing or otherwise using the memory for other purposes during the tnn network run 
    //  will result in undefined behavior.
    Status SetForwardMemory(void* memory);

    // reshape instance with new input shapes
    Status Reshape(const InputShapesMap& inputs);

    // get tnn command queue
    Status GetCommandQueue(void** command_queue);

    // @brief tnn instance network infer, it will wait until all layer infer complete.
    Status Forward();

    ...

    // tnn instance network infer async.
    // device gpu, all layer infer complete will call Callback.
    Status ForwardAsync(Callback call_back);

    // get all input blobs
    Status GetAllInputBlobs(BlobMap& blobs);

    // get all output blobs
    Status GetAllOutputBlobs(BlobMap& blobs);

    // set threads run on cpu 
    virtual Status SetCpuNumThreads(int num_threads);
    ...

    // set input Mat, if input_name is not set, take the first input as default
    Status SetInputMat(std::shared_ptr<Mat> mat,
                       MatConvertParam param,
                       std::string input_name = "");
    
    // get output Mat, if output_name is not set, take the first output as default
    Status GetOutputMat(std::shared_ptr<Mat>& mat,
                        MatConvertParam param = MatConvertParam(),
                        std::string output_name = "", 
                        DeviceType device = DEVICE_ARM, MatType mat_type = NCHW_FLOAT);

};
```

Instance interface instruction：  

- The `Instance` and `Init` interfaces are normally called by the TNN CreateInst interface, used to generate Instance network instances.  
- `GetForwardMemorySize` can get the memory size required for all the blobs of Instance, `SetForwardMemory` is used to pass in external memory. For Instances built in `SHARE_MEMORY_MODE_SET_FROM_EXTERNAL` memory mode, the memory needs to be passed in from the outside, and the actual size of the incoming memory must not be less than the value returned by `GetForwardMemorySize`.  
- The `Reshape` interface supports resetting the input size after the network is successfully constructed. Only the network built with `min_inputs_shape` and `max_inputs_shape` can change the input size during operation. The variable size range is specified by `min_inputs_shape` and `max_inputs_shape`.  
- The `GetCommandQueue` interface supports obtaining the command queue corresponding to the network operation, and the same command queue message is executed sequentially.  
- `GetAllInputBlobs` and `GetAllOutputBlobs` are used to get input and output blobs respectively.  
- `SetCpuNumThreads` can set the number of parallel CPU threads.  
- `Forward` runs a synchronous interface for the network, and `ForwardAsync` runs an asynchronous interface for the network.  
- `SetInputMat` is used to set the input Mat, where MatConvertParam can set the conversion parameters([mat-convert-parameter description](#MatConvertParam-description)). For multi-input networks, it can be distinguished by input_name.  
- `GetOutputMat` is used to obtain the output result and save it in the output Mat. Among them, MatConvertParam can set the conversion parameters([mat-convert-parameter description](#MatConvertParam-description)). For multi-output networks, it can be distinguished by output_name. DeviceType can specify whether the output Mat Memory is built on the CPU or GPU. MatType is applied to set the output Mat data arrangement.   

### 4. core/mat.h

```cpp
class PUBLIC Mat {
public:
    ...

    Mat(DeviceType device_type, MatType mat_type, DimsVector shape_dims, void* data);
    Mat(DeviceType device_type, MatType mat_type, DimsVector shape_dims);
    //empty mat
    Mat(DeviceType device_type, MatType mat_type);

    DEPRECATED("use Mat(DeviceType, MatType, DimsVector, void*) instead")
    Mat(DeviceType device_type, MatType mat_type, void* data) : Mat(device_type, mat_type, {1,0,0,0}, data) {};

    ...
};
```
MatType supports common CV, NLP input and output layouts, and `DeviceType` can be set to CPU and GPU.

```cpp
typedef enum {
    INVALID    = -1,
    //bgr or rgb: uint8
    N8UC3      = 0x00,
    //bgra or rgba: uint8
    N8UC4      = 0x01,
    //gray: uint8
    NGRAY      = 0x10,
    //YUV420SP, YYYYVUVUVU
    NNV21      = 0x11,
    //YUV420SP, YYYYUVUVUV
    NNV12      = 0x12,
    //nchw: float
    NCHW_FLOAT = 0x20,
    // nchw: int32
    NC_INT32 = 0x21,
    ...
} PUBLIC MatType;
```

### 5. core/macro.h
Provide different platform Log macros, different data types maximum and minimum macros, PUBLIC macro definition, and some data pack conversion and other macro definitions.

### 6. core/status.h
`Status`is defined in status.h.

```cpp
enum StatusCode {

    TNN_OK = 0x0,

    // param errcode
    TNNERR_PARAM_ERR        = 0x1000,
    TNNERR_INVALID_NETCFG   = 0x1002,
    ...
}

class PUBLIC Status {
public:
    Status(int code = TNN_OK, std::string message = "OK");

    Status &operator=(int code);

    bool operator==(int code_);
    bool operator!=(int code_);
    operator int();
    operator bool();
    std::string description();

private:
    int code_;
    std::string message_;
}
```
The error message will be returned in `description` interface when Status code is not equal to TNN_OK.

### 7. core/blob.h

```cpp
// @brief BlobDesc blob data info
struct PUBLIC BlobDesc {
    // device_type describes devie cpu, gpu, ...
    DeviceType device_type = DEVICE_NAIVE;
    // data_type describes data precion fp32, in8, ...
    DataType data_type = DATA_TYPE_FLOAT;
    // data_format describes data order nchw, nhwc, ...
    DataFormat data_format = DATA_FORMAT_AUTO;
    // DimsVector describes data dims
    DimsVector dims;
    // name describes the blob name
    std::string name;
    
    std::string description(bool all_message = false);
};

struct PUBLIC BlobHandle {
    void *base            = NULL;
    uint64_t bytes_offset = 0;
};

// @brief Blob tnn data store and transfer interface.
class PUBLIC Blob {
public:
    ...

    //@brief create Blob with blob descript and data handle
    Blob(BlobDesc desc, BlobHandle handle);

    ...
};

```

Blob is composed of `BlobDesc` and `BlobHandle`, where `BlobDesc` describes Blob related structural information, and `BlobHandle` is used to read and store Blob data.

`BlobDesc` contains `device_type`, `data_type`, `data_format`, `dims`, `name` information.

The `dims` describes the blob dimension information, the dims storage size has nothing to do with data_format:  

- The dims size is 2, and the storage corresponds to N, C.  
- The dims size is 4, and the storage size corresponds to N, C, H, W.  
- The dims size is 5, and the storage size corresponds to N, C, D, H, and W.  

The current input and output data types and arrangements of blobs for different platforms are as follows:  

- `ARM`: CPU memory, NC4HW4.  
- `OPENCL`: GPU graphics memory (clImage), NHC4W4. Among which NH is clImage high, C4W4 is clImage wide.  
- `METAL`: GPU video memory (metal), NC4HW4.  
- `HUAWEI_NPU` :CPU memory, NCHW.  
- `X86`: CPU memory, NCHW.  
- `CUDA`: GPU memory, NCHW.  

Among them, the last 4 represents pack 4 and C4 represents the last 1 bit 4 is packed by 4 Cs.  


### 8. utils/mat\_utils.h
```cpp
class PUBLIC MatUtils {
public:
    //copy cpu <-> device, cpu<->cpu, device<->device, src and dst dims must be equal.
    static Status Copy(Mat& src, Mat& dst, void* command_queue);

    //src and dst device type must be same. when param scale_w or scale_h is 0, it is computed as
    // (double)dst.GetWidth() / src.GetWidth() or (double)dst.GetHeight() / src.GetHeight().
    static Status Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue);

    //src and dst device type must be same. when param width or height is 0, it is equal to
    //dst.GetWidth() or dst.GetHeight().
    static Status Crop(Mat& src, Mat& dst, CropParam param, void* command_queue);

    //src and dst device type must be same.
    static Status WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue);

    //src and dst device type must be same.
    static Status CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue);

    //src and dst device type must be same. param top, bottom, left and right must be non-negative.
    static Status CopyMakeBorder(Mat& src, Mat& dst, CopyMakeBorderParam param, void* command_queue);
};
```

interface instruction:  

- `Copy`: Support different DEVICE and CPU Mat data copy, and Mat data copy between the same DEVICE.  
-  `Resize`, `Crop`, `WarpAffine`, `CvtColor`, `CopyMakeBorder` interface behavior is similar to OpenCV, both CPU and GPU support, `src` and `dst` must have the same `DEVICE_TYPE`.

### 9. utils/bfp16\_utils.h
The interface provides the cpu memory conversion tool between fp16 and fp32. 


### 10. utils/blob\_convert.h
```cpp
class PUBLIC BlobConverter {
public:
    explicit BlobConverter(Blob* blob);
    virtual Status ConvertToMat(Mat& image, MatConvertParam param, void* command_queue);
    virtual Status ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue);

    virtual Status ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue);
    virtual Status ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue);

private:
    Blob* blob_;
    std::shared_ptr<BlobConverterAcc> impl_ = nullptr;
};
```

Through `ConvertToMat`, you can import blob data into Mat in Mat format, and `ConvertFromMat` can import Mat data into blob in blob format, and the command_queue can be obtained by the Instance `GetCommandQueue` interface.

It also provides common pre-processing/post-processing: support setting scale/bias parameter and reverse channel adaptation bgr, rgb or other scenarios.

```cpp
struct PUBLIC MatConvertParam {
    std::vector<float> scale = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> bias = {0.0f, 0.0f, 0.0f, 0.0f};
    bool reverse_channel = false;
};
```

#### MatConvertParam description:  
- `reverse_channel`: The default is `false`. If blue channel and red channel of the input mat need to be reversed, the parameter could be set to `true`.  
    * The parameter is only valid for mats of `N8UC3` or `N8UC4` type. For other types of mat, the parameter is ignored.  
    * Both `ConvertFromMat` and `ConvertToMat` procedures support reverse of channel.  
- `scale` and `bias`: The default of scale is `1`, and the default of bias is `0`. The input mat is first multiplied by the scale, and then added with the bias.  
    * All types of mat support scale and bias.  
    * Both `ConvertFromMat` and `ConvertToMat` procedures support scale and bias.  
    * If scale values are all equal to `1`, and bias values are all equal to `0`, or the default scale and bias are used, the scale and bias procedure will be skipped. Otherwise, both numbers of scale values and bias values should be consistent with the input channel.  
    * For multi-channel inputs, the order of scale values and bias values should be consistent with the data format used in model inference. For example, if the model actually uses BGR images to do inference, then scale and bias should follow BGR order for both `ConvertFromMat` and `ConvertToMat` procedures, no matter reverse channel or not. This is also equivalent to first reverse channel, then do scale and bias for `ConvertFromMat`, and first do scale and bias, then reverse channel for `ConvertToMat`.  

### 11. utils/cpu\_utils.h
Provide tools that are related to CPU thread core binding and power saving mode setting.

### 12. utils/data\_type\_utils.h
Provide DataType size and name conversion-related tools.

### 13. utils/dims\_vector\_utils.h
Provide commonly-used blob dims calculation and comparison tools.

### 14. utils/half\_utils.h
The interface provides CPU memory conversion tools between fp32 and fp16.

### 15. utils/string\_utils.h
The interface provides conversion from uchar string to std::string, which is mainly used for TNN model memory input.

### 16. version.h
Build version information.
