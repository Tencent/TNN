# API documentation

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

Compatibility of different API versions follows[Semantic Version 2.0.0](https://semver.org/lang/zh-CN/) rules.

## II. the API directory structure

```bash
.
└── tnn
    ├── core
    │ ├── blob.h # responsible for data transmission
    │ ├── common.h # define common structure
    │ ├── instance.h # network instance
    │ ├── macro.h # common macro definitions
    │ ├── status.h # interface status
    │ └── tnn.h # model parsing
    ├── utils
    │ ├── blob_converter.h # tools of blob input and output data
    │ ├── cpu_utils.h # CPU performance optimization tool
    │ ├── data_type_utils.h # network data type parsing tool
    │ ├── dims_vector_utils.h # blob size calculation tool
    │ └── half_utils.h # fp16 conversion tool
    └── version.h # compile and build information
```


## III. API introduction

### 1. Model parsing

> The first step is model parsing. The TNN class defined in tnn.h is responsible for model parsing.

```cpp
class PUBLIC TNN {
public:
    ...

    Status Init (ModelConfig & config);

    // denit tnn implement, release model interpreter.
    Status DeInit ();

    // add output to the model.
    // if output_name of blob not found, then search output_index of layer.
    Status AddOutput (const std :: string & output_name, int output_index = 0);
    ...
};
```

Common interfaces of TNN:
- Init interface: Responsible for importing and parsing model data, which needs to be configured and passed into ModelConfig.
- DeInit interface: Responsible for the release of tnn implementation, the default destructor can be automatically released.
- AddOutput interface: Support adding model outputs; the model output can be defined arbitrarily as the output of any intermediate layer.

> The TNN Init interface passed into ModelConfig is defined in the common.h header file.

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
    std :: vector <std :: string> params;
};
```

ModelConfig parameter description:
- `model_type`: The current open source version of TNN only supports two model formats:` MODEL_TYPE_TNN` and `MODEL_TYPE_NCNN`.
- `params`: The TNN model needs to pass in the content of the proto file and the path of the model file. The NCNN model needs to pass in the content of the param file and the path of the bin file.

> model_type enumeration type ModelType is defined in common.h

```cpp
typedef enum {
    MODEL_TYPE_TNN = 0x0001,
    MODEL_TYPE_NCNN = 0x0100,
    ...
} ModelType;
```

All status codes are returned to the TNN interface. Among them, `Status` can display status codes and error messages.` Status` is defined in the status.h header file.

```cpp
enum StatusCode {

    RPD_OK = 0x0,

    // param errcode
    RPDERR_PARAM_ERR = 0x1000,
    RPDERR_INVALID_NETCFG = 0x1002,
    ...
}

class PUBLIC Status {
public:
    Status (int code = RPD_OK, std :: string message = "OK");

    Status & operator = (int code);

    bool operator == (int code_);
    bool operator! = (int code_);
    operator int ();
    operator bool ();
    std :: string description ();

private:
    int code_;
    std :: string message_;
}
```

When the Status code is not RPD_OK, error description information can be returned through the `description` interface.

### 2. Network instance construction

> The second step of using TNN is to build a network instance. You can build a network instance through the TNN `CreateInst` interface.

```cpp
class PUBLIC TNN {
public:
    ...

    // create tnn network instance with network config and inputs shape.
    // if inputs shape not set, use default from model.
    std :: shared_ptr <Instance> CreateInst (
        NetworkConfig & config, Status & status,
        InputShapesMap inputs_shape = InputShapesMap ());

private:
    std :: shared_ptr <TNNImpl> impl_;
};
```

Need to configure and pass in NetworkConfig, this interface supports to re-initialize the network input size.

NetworkConfig is defined in common.h:

```cpp
// @brief Config used to create tnn instance, config
// device type, network type and share memory mode.
struct PUBLIC NetworkConfig {
    // device type default cpu
    DeviceType device_type = DEVICE_CPU;

    // device id default 0
    int device_id = 0;

    // blob data format decided by device
    DataFormat data_format = DATA_FORMAT_AUTO;

    // network type default internal
    NetworkType network_type = NETWORK_TYPE_DEFAULT;

    // raidnet instances not share memory with others
    ShareMemoryMode share_memory_mode = SHARE_MEMORY_MODE_DEFAULT;

    // dependent library path
    std :: vector <std :: string> library_path;
};
```


NetworkConfig parameter description:
- `device_type`: The default is DEVICE_CPU, which does not include platform-specific acceleration instructions.
    * Android uses DEVICE_ARM, DEVICE_OPENCL for acceleration
    * iOS uses DEVICE_ARM, DEVICE_OPENCL to accelerate.
- `device_id`: The default is 0, which supports multiple devices selection by device_id, not mandatory for the mobile platform.
- `data_format`: By default, tnn automatically selects a blob data format for acceleration. You can set a specific blob data format through this parameter
- `network_type`: Support for building tnn custom networks and third-party networks. The current open source version only supports building tnn networks.
- `share_memory_mode`: tnn instance memory sharing mode
    * `SHARED_MEMORY_MODE_DEFAULT`: only supports memory sharing between different blobs of the same instance
    * `SHARE_MEMORY_MODE_SHARE_ONE_THREAD`: support memory sharing of different instances of the same thread
    * `SHARE_MEMORY_MODE_SET_FROM_EXTERNAL`: support for instance memory to be passed in from outside, the sharing method is determined by the function caller, sharing among threads needs to dealwith synchronization issues, and memory allocation and release require maintenance on the caller.
- `library_path`: support external dependent library loading, this parameter needs to be configured if the iOS metal kernel library is not placed in the app default path.

The NetworkConfig enumeration types: DeviceType, DataFormat, NetworkType, and ShareMemoryMode are all defined in common.h.

```cpp
typedef enum {
    // decided by device
    DATA_FORMAT_AUTO = -1,
    DATA_FORMAT_NCHW = 0,
    DATA_FORMAT_NHWC = 1,
    DATA_FORMAT_NHWC4 = 2,
    DATA_FORMAT_NC4HW4 = 3,
    DATA_FORMAT_NCDHW = 4,
    DATA_FORMAT_NHC4W4 = 5,
} DataFormat;

typedef enum {
    NETWORK_TYPE_DEFAULT = 0,
    ...
} NetworkType;

typedef enum {
    DEVICE_CPU = 0x0000,
    DEVICE_X86 = 0x0010,
    DEVICE_ARM = 0x0020,
    DEVICE_OPENCL = 0x1000,
    DEVICE_METAL = 0x1010,
    ...
} DeviceType;

typedef enum {
    // default
    SHARE_MEMORY_MODE_DEFAULT = 0,
    // same thread tnn instance share blob memory
    SHARE_MEMORY_MODE_SHARE_ONE_THREAD = 1,
    // set blob memory from external, different thread share blob memory need
    // synchronize
    SHARE_MEMORY_MODE_SET_FROM_EXTERNAL = 2
} ShareMemoryMode;

typedef enum {
    MODEL_TYPE_TNN = 0x0001,
    MODEL_TYPE_NCNN = 0x0100,
    ...
} ModelType;
```


### 3. Network instance running

> the third step of using TNN is network instance. Through the Instance interface, it could set the network input data, do the inference of the network, and obtain network output data.

To set input data and get output data, you need to get network input and output blobs first. You can get all network input blobs through the `GetAllInputBlobs` interface, and you can get all network output blobs through the` GetAllOutputBlobs` interface.

```cpp
class PUBLIC Instance {
public:
    ...

    // get all input blobs
    Status GetAllInputBlobs (BlobMap & blobs);

    // get all output blobs
    Status GetAllOutputBlobs (BlobMap & blobs);

    ...
};
`` `

The network runs through the `Forward` and` ForwardAsync` interfaces, `ForwardAsync` is an asynchronous interface.

```cpp
class PUBLIC Instance {
public:
    ...

    // get tnn command queue
    Status GetCommandQueue (void ** command_queue);

    // @brief tnn instance network infer, it will wait until all layer infer complete.
    Status Forward ();

    ...

    // tnn instance network infer async.
    // device gpu, all layer infer complete will call Callback.
    Status ForwardAsync (Callback call_back);

    // get all input blobs
    Status GetAllInputBlobs (BlobMap & blobs);

    ...
};
```

data_format and data_type of input blobs may be different on different platforms, and setting and obtaining data for GPU blob need to write GPU related code. The API provides a simple tool BlobConverter for input blob data setting and output blob data acquisition, defined in blob_converter.h.

```cpp
class PUBLIC BlobConverter {
public:
    explicit BlobConverter (Blob * blob);
    virtual Status ConvertToMat (Mat & image, MatConvertParam param, void * command_queue);
    virtual Status ConvertFromMat (Mat & image, MatConvertParam param, void * command_queue);

    virtual Status ConvertToMatAsync (Mat & image, MatConvertParam param, void * command_queue);
    virtual Status ConvertFromMatAsync (Mat & image, MatConvertParam param, void * command_queue);

private:
    Blob * blob_;
    std :: shared_ptr <BlobConverterAcc> impl_ = nullptr;
};
```

Through `ConvertToMat`, you can import blob data into Mat in Mat format, and` ConvertFromMat` can import Mat data into blob in blob format, and the corresponding `command_queue` of the interface can be obtained through the Instance `GetCommandQueue` interface.


Mat is defined in blob_converter.h,

```cpp
class PUBLIC Mat {
public:
    ...

    Mat (DeviceType device_type, MatType mat_type, void * data);
    Mat (DeviceType device_type, MatType mat_type, DimsVector shape_dims);
    ...
};
```

MatType supports commonly used CV input and output layouts, and `DeviceType` can be set to CPU and GPU.

```cpp
typedef enum {
    INVALID = -1,
    N8UC3 = 0x00,
    N8UC4 = 0x01,
    NGRAY = 0x10,
    NNV21 = 0x11,
    NNV12 = 0x12,
    NCHW_FLOAT = 0x20,
} PUBLIC MatType;
```

At the same time, it supports common pre-processing, post-processing, scale setting, bias settings and reverse channel reverse for bgr/rgb format.

```cpp
struct PUBLIC MatConvertParam {
    std :: vector <float> scale = {1.0f, 1.0f, 1.0f, 1.0f};
    std :: vector <float> bias = {0.0f, 0.0f, 0.0f, 0.0f};
    bool reverse_channel = false;
};
```

For detailed instructions, you can read the demo documentation.

The input and output blob is defined in the blob.h header file

```cpp
// @brief BlobDesc blob data info
struct PUBLIC BlobDesc {
    // deivce_type describes devie cpu, gpu, ...
    DeviceType device_type = DEVICE_CPU;
    // data_type describes data precion fp32, in8, ...
    DataType data_type = DATA_TYPE_FLOAT;
    // data_format describes data order nchw, nhwc, ...
    DataFormat data_format = DATA_FORMAT_AUTO;
    // DimsVector describes data dims
    DimsVector dims;
    // name describes the blob name
    std :: string name;
};

struct PUBLIC BlobHandle {
    void * base = NULL;
    uint64_t bytes_offset = 0;
};

// @brief Blob tnn data store and transfer interface.
class PUBLIC Blob {
public:
    ...

    // @ brief create Blob with blob descript and data handle
    Blob (BlobDesc desc, BlobHandle handle);

    ...
};

```

Blob is currently mainly composed of `BlobDesc` and` BlobHandle`, where `BlobDesc` describes blob-related structural information, and` BlobHandle` is used to read and store blob data.

`BlobDesc` is used to describe device_type, data_type, data_format, dims, name information.
Where data_type is defined in the common.h header file

```cpp
typedef enum {
    DATA_TYPE_FLOAT = 0,
    DATA_TYPE_HALF = 1,
    DATA_TYPE_INT8 = 2,
    DATA_TYPE_INT32 = 3
} DataType;
```

dims describes the blob dimension information, the dims storage has nothing to do with data_format:
- The dims size is 4, and the storage corresponds to N, C, H, W.
- The dims size is 5, and the storage corresponds to N, C, D, H, and W.

The current input and output data types and format for different platforms are as follows:

- `ARM`: CPU memory, NC4HW4.
- `OPENCL`: GPU memory (clImage), NHC4W4. NH is the height of clImage, C4W4 is width of clImage.
- `METAL`: GPU memory (metal), NC4HW4. The last 4 represents pack 4, and C4 represents packed by 4 channels.

### 4. Other supplementary notes

Supplementary explanations of Instance interface:

```cpp
class PUBLIC Instance {
public:

    ...

    // return memory bytes required for forward
    Status GetForwardMemorySize (int & memory_size);

    // set memory to tnn instance. if success, return status code zero.
    // only instance created with SHARE_MEMORY_MODE_SET_FROM_EXTERNAL can be set from external.
    // the memory size need> = GetForwardMemorySize ().
    // releasing or otherwise using the memory for other purposes during the tnn network run
    // will result in undefined behavior.
    Status SetForwardMemory (void * memory);

    // reshape instance with new input shapes
    Status Reshape (const InputShapesMap & inputs);
    ...
    // set threads run on cpu
    virtual Status SetCpuNumThreads (int num_threads);
    ...
};
```


Some instructions about Instance:
- TNN Instance supports a multi-thread setting, and the number of parallel CPU threads can be set through `SetCpuNumThreads`.
- The TNN `Reshape` interface supports resetting network input and output. The current implementation of` Reshape` does not reallocate memory, so the size passed to `Reshape` must not be greater than the initial network size.
- For Instances built in `SHARE_MEMORY_MODE_SET_FROM_EXTERNAL` memory mode, the required memory needs to be passed in through` SetForwardMemory`, the required memory size can be obtained through `GetForwardMemorySize`, and the actual size of the incoming memory should not be less than the value returned by` GetForwardMemorySize`.

Additional notes on other `Utils` interfaces:
- `half_utils.h`: the internal interface provides conversion tools between fp32 and fp16 for CPU memory.
- `dims_vector_utils.h`: The internal interface provides common tools for blob dims calculation and comparison.
- `data_type_utils.h`: the internal interface provides tools related to dataType size and name conversion.
- `cpu_utils.h`: the internal interface provides tools related to CPU thread core binding and power saving mode setting.
- `macro.h`: provide log macros for different platforms, macros for maximum and minimum values of different data types, PUBLIC macro definitions, and some macros for data pack conversion.