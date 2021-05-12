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
    - 若要支持ARMv8.2编译，ndk版本版本至少为r18b
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

编译完成后，在当前目录的`release`目录下生成对应的`armeabi-v7a`库，`arm64-v8a`库和`include`头文件。<font color="#dd0000">如果是编译成静态库，集成链接需添加`-Wl,--whole-archive tnn -Wl,--no-whole-archive`</font>。

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
2）编辑`build_aarch_linux.sh` 或 `build_armhf_linux.sh` 修改配置选项 
```
 SHARED_LIB="ON"                # ON表示编译动态库，OFF表示编译静态库
 ARM="ON"                       # ON表示编译带有Arm CPU版本的库
 OPENMP="ON"                    # ON表示打开OpenMP
 OPENCL="OFF"                   # ON表示编译带有Arm GPU版本的库
 RKNPU="OFF"                    # ON表示编译带有RKNPU版本的库
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
./build_aarch_linux.sh
```
RKNPU : 运行前需要下载DDK, 并放到指定文件夹。具体请参考:
[FAQ](../faq.md#如何创建rknpu编译环境)如何创建RKNPU编译环境?

## 四、Linux 环境编译
### 1.环境要求
依赖库
  - cmake (使用3.11版本及以上)
  - 网络访问

### 2.编译步骤
1）切换到脚本目录
```
cd <path_to_tnn>/scripts
```
2) 执行编译脚本
  - 编译不带openvino的版本
```
./build_linux_native.sh
```
  - 编译带openvino的版本
```
./build_x86_linux.sh
```
注意：openvino只能编译成64位的库，且cmake版本必须要求3.13以上

## 五、Linux CUDA库编译
### 1.环境要求
#### 依赖库
  - cmake (使用3.8及以上版本）
  - CUDA (使用10.2及以上版本)

#### TensorRT配置
  - 下载TensorRT(>=7.1) <https://developer.nvidia.com/nvidia-tensorrt-7x-download>
  - 配置环境变量 `export TENSORRT_ROOT_DIR=<TensorRT_path>`

#### CuDNN配置
  - 下载CuDNN(>=8.0) <https://developer.nvidia.com/rdp/cudnn-download>
  - 配置环境变量 `export CUDNN_ROOT_DIR=<CuDNN_path>`

### 2.编译步骤
1）切换到脚本目录
```
cd <path_to_tnn>/scripts
```
2) 执行编译脚本
```
./build_cuda_linux.sh
```

## 六、Windows 环境编译
### 1.环境要求
依赖库
  - Visual Studio (2017 及更高版本)
  - cmake (把3.11及以上版本cmake加入环境变量或使用 Visual Studio 自带cmake)
  - ninja (编译速度更快，可以使用choco安装)

### 2.编译步骤
打开 `x64 Native Tools Command Prompt for VS 2017/2019`，如果想要编译32位的库，打开 `x86 Native Tools Command Prompt for VS 2017/2019`
1) 切换到脚本目录
```
cd <path_to_tnn>/scripts
```
2) 执行编译脚本
  - 编译不带openvino的版本
```
.\build_msvc_naive.bat
```
  - 编译带openvino的版本
```
.\build_msvc.bat
```
openvino只能编译成64位的库，更多编译问题请参考 [FAQ](openvino.md)

## 七、Windows CUDA 环境编译
### 1.环境要求
依赖库
  - Visual Studio (2017 及更高版本)
  - cmake (把3.11及以上版本cmake加入环境变量或使用 Visual Studio 自带cmake)
  - CUDA (使用10.2及以上版本)

#### TensorRT配置
  - 下载TensorRT(>=7.1) <https://developer.nvidia.com/nvidia-tensorrt-7x-download>
  - 在脚本文件 *build_cuda_msvc.bat* 中修改 `set TENSORRT_ROOT_DIR=<TensorRT_path>`

#### CuDNN配置
  - 下载CuDNN(>=8.0) <https://developer.nvidia.com/rdp/cudnn-download>
  - 在脚本文件 *build_cuda_msvc.bat* 中修改 `set CUDNN_ROOT_DIR=<CuDNN_path>`

### 2.编译步骤
打开 `x64 Native Tools Command Prompt for VS 2017/2019` 或配置了cmake环境变量的 `cmd`
1) 切换到脚本目录
```
cd <path_to_tnn>/scripts
```
2) 执行编译脚本
```
.\build_cuda_msvc.bat
```

## 八、Macos 环境编译
### 1.环境要求
依赖库
  - cmake 3.11 以上版本 
  - xcode command line tools (需提前在应用商店安装好Xcode，然后再命令行执行xcode-select --install )
  - automake, libtool (可通过brew安装，指令是brew install libtool, brew install automake)
  - 网络访问

### 2.编译步骤
1）切换到脚本目录
```
cd <path_to_tnn>/scripts
```
2）执行编译脚本
```
./build_macos.sh
```

## 编译参数option说明

|Option|默认值|说明|
|------|:---:|----|
|TNN_CPU_ENABLE| ON | 代码source/device/cpu编译开关，实现全部为c++代码，不包含特定CPU加速指令。|
|TNN_X86_ENABLE| OFF | 代码source/device/x86编译开关, 当前适配openvino实现，后续会迁入更多加速代码实现。|
|TNN_ARM_ENABLE| OFF | 代码source/device/arm编译开关，代码包含neon加速指令, 且部分实现了int8加速。|
|TNN_ARM82_ENABLE| OFF | 代码source/device/arm/acc/compute_arm82编译开关，代码包含fp16指令加速。|
|TNN_METAL_ENABLE| OFF | 代码source/device/metal编译开关，代码包含metal加速指令。|
|TNN_OPENCL_ENABLE| OFF | 代码source/device/opencl编译开关，代码包含opencl加速指令。|
|TNN_CUDA_ENABLE| OFF | 代码source/device/cuda编译开关，当前适配TensorRT实现，后续会迁入更多加速代码实现。|
|TNN_DSP_ENABLE| OFF | 代码source/device/dsp编译开关，当前适配snpe实现。|
|TNN_ATLAS_ENABLE| OFF | 代码source/device/atlas编译开关，当前适配华为atlas加速框架。|
|TNN_HUAWEI_NPU_ENABLE| OFF | 代码source/device/huawei_npu编译开关，当前适配HiAI加速框架。|
|TNN_RK_NPU_ENABLE| OFF | 代码source/device/rknpu编译开关，当前适配rknpu_ddk加速框架。|
|TNN_SYMBOL_HIDE| ON | 加速库符号隐藏，release发布默认非public接口符号不可见。|
|TNN_OPENMP_ENABLE| OFF | OpenMP开关，控制是否打开openmp加速。|
|TNN_BUILD_SHARED| ON | 动态库编译开关，关闭则编译静态库。|
|TNN_TEST_ENABLE| OFF | test代码编译开关|
|TNN_UNIT_TEST_ENABLE| OFF | unit test编译开关，打开unit test编译开关会自动打开TNN_CPU_ENABLE开关，作为测试基准。|
|TNN_PROFILER_ENABLE| OFF | 性能调试开关，打开后会打印更多性能信息，仅用于调试。|
|TNN_QUANTIZATION_ENABLE| OFF | 量化工具编译开关|
|TNN_BENCHMARK_MODE| OFF | benchmark开关，打开后支持model weights文件为空，可自动生成数据。|
|TNN_ARM82_SIMU| OFF | ARM82仿真开关，需要和TNN_ARM82_ENABLE同时打开，打开后可以在普通CPU上运行half实现代码。|

