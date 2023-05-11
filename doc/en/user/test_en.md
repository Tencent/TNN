# Model Test  

[中文版本](../../cn/user/test.md)

## I. Compile
Refer to [Install and Compile](./compile_en.md).
Enable the test options：
* `TNN_TEST_ENABLE:BOOL=ON`  
* set corresponding devices，such as `TNN_OPENCL_ENABLE`, `TNN_ARM_ENABLE`, `TNN_HUAWEI_NPU_ENABLE`
* After the compilation is completed, an executable file 'test/ TNNTest' will be generated in the build directory, which can be run directly in Linux, Android ADB and other environments

## II. Usage 
### 1. Command
```
TNNTest
Required Parameters：
  -mp Model or Model Proto Position
    - when model type is TNN Proto + Model, fill in absolute or relative path of TNN Proto file，e.g.：-mp ./model/SqueezeNet/squeezenet_v1.1.tnnproto
    - when model type is TorchScript, fill in absolute or relative path of TorchScript file
    - when model type is HUAWEI ATLAS OM, fill in absolute or relative path of OM file
  -dt Target Device Type
    - NAIVE - run on X86 or arm CPU，NAIVE mode ops are written by C++ without any optimization tech, NAIVE is for correctness checking
    - X86 - run on X86_64 CPU
    - ARM - run on ARM V7/V8 CPU
    - CUDA - run on NVidia GPU
    - OPENCL - run on OPENCL available GPU
    - METAL - run METAL supported APPLE GPU
    - HUAWEI_NPU - run on HUAWEI Hiai NPU
    - RKNPU - run on RockChip RK NPU
    - APPLE_NPU - run on APPLE NPU
    - ZIXIAO - run on ZIXIAO NPU
    - ATLAS - run on HUAWEI ATLAS NPU
  -nt TNN Network Type
    - Default - when target device is X86, ARM, CUDA etc, TNN will automatically select network, default is OK
    - COREML - run APPLE CoreML
    - HUAWEI_NPU - run HUAWEI Hiai NPU
    - OPENVINO - run OpenVINO
    - RKNPU - run RockChip RK NPU
    - SNPE - run Qualcomm SNPE
    - TRT - run NVidia TensorRT
    - TORCH - run Pytorch TorchScript Network
    - ZIXIAO - run ZIXIAO Network
  -op Output File Location
    - blank(default) - end TNNTest without saving Output, not recommended
    - ${YOUR_OUTPUT_NAME}.txt - absolute path for output.txt file，TNNTest will write all network results to this file

```
TNNTest
Frequently Used Parameters：
  -pr Network Precision
    - AUTO(default) - automatically select precision depends on target device, may be either float32 or float16 depending on target device type
    - HIGH - set precision to float32
    - LOW - set precision to float16
  -wc Warmup Iterations (not timed)
    - 0(default) - TNNTest will start timing on the first forward by default，the first several Forward Calls might be slow on some devices
    - N - run N extra forward steps before timing start
  -ic Counted Iterations (timed)
    - 1(default) - TNNTest will run 1 forward pass by default, 1 is recommended in result checking
    - N - run N timed forward steps, TNNTest will record MIN/MAX/AVG forward time.
  -dl Device Numbers
    - 0(default) - TNNTest will run on device 0 by default
    - Specified Values - specify one or multiple devices for TNNTest to run on, separated by comma，e.g.: -dl "0,1,2,3,4,5"
  -is Input Shapes
    - blank(default) - when mode type is TNN Proto, TNNTest will read default input shapes from TNN Proto, for other model types, -is is required
    - Specified Values - specify one or multiple input shapes for TNNTest to run on, separated by semi-colon，e.g.: "in_0:1,3,512,512;in_2:1,3,256,256;"
  -it Input DataTypes
    - blank(default) - when mode type is TNN Proto, TNNTest will read default input data types from TNN Proto, for other model types, -it is required, input type is float32 by default in most cases
    - Specified Values - specify one or multiple input data types for TNNTest to run on, separated by semi-colon，e.g.: "in_0:0;;in_2:3;"
    - numbers and datatypes - 0->float32; 1->float16; 2->int8; 3->int32; 4->bfp16; 5->int64; 6->uint32; 8->uint8

The test will output the timing info as：time cost: min = xx   ms  |  max = xx   ms  |  avg = xx   ms

It can also be used as a benchmark tool. When you use it, you need to formulate wc> = 1, because the first run will prepare memory, context, etc.,which increases time consumption

```
### 2.  NPU
The HiAI so libraries needs to be pushed to the phone，and which 

## II. Example
### 1. CUDA TNNTest Example
```
./scripts/build_cuda_linux/test/TNNTest \
    -mp ./model/SqueezeNet/squeezenet_v1.1.tnnproto \
    -pr HIGH \
    -is "data:1,3,224,224;" \
    -dt CUDA \
    -wc 0 \
    -ic 1 \
    -op ./squeezenet_result.txt
