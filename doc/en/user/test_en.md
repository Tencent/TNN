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
required parameters： 
    -mp path to model and proto(The proto and model should have the same prefix in the same folder)
    -dt device type (ARM, OPENCL, HUAWEI_NPU)
optional parameters：
    -nt network type（default naive， npu needs to be specified as -nt HUAWEI_NPU）
    -op path of the output  
    -ic loop counter
    -wc warmup counter 
    -dl device list 
    -ip input 
    -it input type，default is NCHW float
    -th CPU thread number 

The test will output the timing info as：time cost: min = xx   ms  |  max = xx   ms  |  avg = xx   ms

It can also be used as a benchmark tool. When you use it, you need to formulate wc> = 1, because the first run will prepare memory, context, etc.,which increases time consumption
```
### 2.  NPU
The HiAI so libraries needs to be pushed to the phone，and which 
