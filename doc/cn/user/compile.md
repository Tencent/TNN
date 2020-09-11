# 从源代码编译

[English Version](../../en/user/compile_en.md)

## 一、iOS库编译
### 1. 编译环境要求
  - Mac系统, Xcode IDE
  - cmake（使用3.1及以上版本）

### 2. 编译步骤
1）切换到脚本目录
```
cd <path_to_tnn>/scripts
```
2）执行编译脚本
```
./build_ios.sh
```
编译过程中如果出现xcrun、metal或metallib命令找不到，可尝试如下命令。
```
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer/
```
编译完成后，在目录`platforms/ios`下产生`tnn.framework`库和`tnn.bundle`资源
3）添加到工程  

  - 在iOS app工程的根目录中添加`tnn.framework`库和`tnn.bundle`资源;  
  - 在app Xcode工程的设置中找到`Build Setting -> Linking -> Other Linker Flags`选项;  
  - 添加`-force_load "$(path_to_tnn)/tnn.framework/tnn"`;  

### 3. 限制说明

当前编译出的`tnn.framework`支持iOS设备上跑CPU和GPU，在Mac设备上当前仅支持跑GPU，CPU的支持在后续版本迭代中支持。

## 二、Android库编译
### 1. 环境要求
#### 依赖库
  - cmake（使用3.6及以上版本）

#### NDK配置
  - 下载ndk版本(>=15c)  <https://developer.android.com/ndk/downloads>
  - 配置环境变量 `export ANDROID_NDK=<ndk_path>`
### 2. 命令依赖
centos:
```shell script
yum install attr.x86_64
```
ubuntu:
```shell script
sudo apt-get install attr
```
### 3. 编译步骤
1）切换到脚本目录
```
cd <path_to_tnn>/scripts
```
2）编辑`build_android.sh`修改配置选项 
```
 ABIA32="armeabi-v7a with NEON"
 ABIA64="arm64-v8a"
 STL="c++_static"
 SHARED_LIB="ON"                # ON表示编译动态库，OFF表示编译静态库
 ARM="ON"                       # ON表示编译带有Arm CPU版本的库
 OPENMP="ON"                    # ON表示打开OpenMP
 OPENCL="ON"                    # ON表示编译带有Arm GPU版本的库
 HUAWEI_NPU="ON"                # ON表示编译带有Arm GPU NPU版本的库
 SHARING_MEM_WITH_OPENGL=0      # 1表示OpenGL的Texture可以与OpenCL共享
```
华为NPU PS: 
运行前需要下载DDK, 并放到指定文件夹。 或是用脚本直接下载具体请参考:
[FAQ](../faq.md)如何创建华为NPU编译环境? 

3）执行编译脚本
```
./build_android.sh
```

编译完成后，在当前目录的`release`目录下生成对应的`armeabi-v7a`库，`arm64-v8a`库和`include`头文件。

## 三、ARM Linux跨平台交叉编译

### 1. 环境要求
#### 依赖库
  - cmake（使用3.1及以上版本）
  - 交叉编译需要安装编译工具链
  - ubuntu: aarch64: sudo apt-get install g++-aarch64-linux-gnu  gcc-aarch64-linux-gnu
            arm32hf: sudo apt-get install g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf
  - other linux: 下载arm toolchain: https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads
### 2. 编译步骤
1）切换到脚本目录
```
cd <path_to_tnn>/scripts
```
2）编辑`build_aarch_linux.sh` 或 `build_arm32hf_linux.sh` 修改配置选项 
```
 SHARED_LIB="ON"                # ON表示编译动态库，OFF表示编译静态库
 ARM="ON"                       # ON表示编译带有Arm CPU版本的库
 OPENMP="ON"                    # ON表示打开OpenMP
 OPENCL="OFF"                   # ON表示编译带有Arm GPU版本的库
 #ARM64:
 CC=aarch64-linux-gnu-gcc       # 指定C编译器
 CXX=aarch64-linux-gnu-g++      # 指定C++编译器
 TARGET_ARCH=aarch64            # 指定指令架构
 #ARM32HF:
 CC=arm-linux-gnueabihf-gcc       
 CXX=arm-linux-gnueabihf-g++      
 TARGET_ARCH=arm
```
3）执行编译脚本
```
./build_arm_linux.sh
```

## 编译参数option说明

|Option|默认值|说明|
|------|:---:|----|
|TNN_CPU_ENABLE| OFF | 代码source/device/cpu编译开关，代码仅用用于调试以及UnitTest基准测试，实现全部为c++代码，不包含特定CPU加速指令。|
|TNN_X86_ENABLE| OFF | 代码source/device/x86编译开关, 当前适配openvino实现，后续会迁入更多加速代码实现。|
|TNN_ARM_ENABLE| OFF | 代码source/device/arm编译开关，代码包含neon加速指令, 且部分实现了int8加速。|
|TNN_METAL_ENABLE| OFF | 代码source/device/metal编译开关，代码包含metal加速指令。|
|TNN_OPENCL_ENABLE| OFF | 代码source/device/opencl编译开关，代码包含opencl加速指令。|
|TNN_CUDA_ENABLE| OFF | 代码source/device/cuda编译开关，代码包含cuda加速指令, 当前仅迁移了小部分实现。|
|TNN_DSP_ENABLE| OFF | 代码source/device/dsp编译开关，当前适配snpe实现。|
|TNN_ATLAS_ENABLE| OFF | 代码source/device/atlas编译开关，当前适配华为atlas加速框架。|
|TNN_HUAWEI_NPU_ENABLE| OFF | 代码source/device/huawei_npu编译开关，当前适配HiAI加速框架。|
|TNN_SYMBOL_HIDE| ON | 加速库符号隐藏，release发布默认非public接口符号不可见。|
|TNN_OPENMP_ENABLE| OFF | OpenMP开关，控制是否打开openmp加速。|
|TNN_BUILD_SHARED| ON | 动态库编译开关，关闭则编译静态库。|
|TNN_TEST_ENABLE| OFF | test代码编译开关|
|TNN_UNIT_TEST_ENABLE| OFF | unit test编译开关，打开unit test编译开关会自动打开TNN_CPU_ENABLE开关，作为测试基准。|
|TNN_PROFILER_ENABLE| OFF | 性能调试开关，打开后会打印更多性能信息，仅用于调试。|
|TNN_QUANTIZATION_ENABLE| OFF | 量化工具编译开关|
|TNN_BENCHMARK_MODE| OFF | benchmark开关，打开后支持model weights文件为空，可自动生成数据。|
