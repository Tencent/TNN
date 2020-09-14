# Compile

[中文版本](../../cn/user/compile.md)

## I. Compile (iOS)
### 1. Environment requirements
  - Mac, Xcode IDE
  - cmake（version 3.1 or higher）

### 2. Steps
1）switch to 'scripts' dir
```
cd <path_to_tnn>/scripts
```
2）execute the build script
```
./build_ios.sh
```
If the `xcrun`, `metal` or `metallib` commands cannot be found during the compilation, try the following commands.
```
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer/
```
After the compilation, the `tnn.framework` library and` tnn.bundle` resources are generated under the directory `platforms/ios`
3）Add to the project  

  - Add `tnn.framework` library and` tnn.bundle` resource under the root directory of iOS app project;
  - Find the `Build Setting -> Linking-> Other Linker Flags` option in the settings of the app Xcode project;
  - Add `-force_load" $ (SRCROOT) /tnn.framework "` to it;

### 3. Restrictions

Currently, the compiled `tnn.framework` supports running on CPU and GPU of iOS devices, but only supports running on GPU of Mac. The support for Mac CPU will come in future TNN updates.

## II. Compile (Android)
### 1. Environment requirements
#### Dependency
  - cmake（version 3.6 or higher）

#### NDK configuration
  - Download Android NDK (version>=15c)  <https://developer.android.com/ndk/downloads>
  - Configure the NDK path in env `export ANDROID_NDK=<ndk_path>`

### 2. Compile
1）switch to 'scripts' dir
```
cd <path_to_tnn>/scripts
```
2）edit `build_android.sh` to config the building options 
```
 ABIA32="armeabi-v7a with NEON"
 ABIA64="arm64-v8a"
 STL="c++_static"
 SHARED_LIB="ON"                # ON for dynamic lib，OFF for static lib
 ARM="ON"                       # ON to build for ARM CPU
 OPENMP="ON"                    # ON to build for OpenMP
 OPENCL="ON"                    # ON to build for GPU
 HUAWEI_NPU="ON"                # ON to enable HUAWEI NPU
 SHARING_MEM_WITH_OPENGL=0      # 1 for sharing OpenGL texture with openCL
```

  Huawei NPU PS: 
    You need to download the DDK library files and copy them to the specified directory. You could use a script to do.
    See:
    HuaweiNPU Compilation Prerequisite in [FAQ](../faq_en.md).
    
    
    
3）execute the building script
```
./build_android.sh
```
After the compilation is completed, the corresponding `armeabi-v7a` library, the` arm64-v8a` library and the `include` header file are generated in the` release` directory of the current directory.

## III. Cross-Compile in Linux

### 1. Environment requirements
#### Dependencies

  - cmake（version 3.1 or higher）
  - Install arm toolchain
  - ubuntu: aarch64: sudo apt-get install g++-aarch64-linux-gnu  gcc-aarch64-linux-gnu
            arm32hf: sudo apt-get install g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf
  - other linux: download toolchains from https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads

### 2. Compilation Steps
1）switch to 'scripts' dir
```
cd <path_to_tnn>/scripts
```
2）edit `build_arm_linux.sh` or `build_arm32hf_linux.sh` to config the building options  
```
 SHARED_LIB="ON"                # ON for dynamic lib，OFF for static lib
 ARM="ON"                       # ON to build for ARM CPU
 OPENMP="ON"                    # ON表示打开OpenMP
 OPENCL="OFF"                   # ON to build for GPU
 #for arm64:
 CC=aarch64-linux-gnu-gcc       # set compiler for aarch64 C
 CXX=aarch64-linux-gnu-g++      # set compiler for aarch64 C++
 TARGET_ARCH=aarch64
 #for arm32hf:
 CC=arm-linux-gnueabihf-gcc       # set compiler for aarch64 C
 CXX=arm-linux-gnueabihf-g++      # set compiler for aarch64 C++
 TARGET_ARCH=arm
```
3）execute the building script
```
./build_arm_linux.sh
```

## Description for build options 

|Option|Default|Description|
|------|:---:|----|
|TNN_CPU_ENABLE| OFF | Code source/device/cpu compilation switch, the code is only used for debugging and UnitTest benchmark test, the implementation is all c ++ code, does not contain specific CPU acceleration instructions.|
|TNN_X86_ENABLE| OFF | The code source/device/x86 compilation switch is currently adapted to the openvino implementation, and more accelerated code implementation will be moved in later.|
|TNN_ARM_ENABLE| OFF | Code source/device/arm compilation switch, the code contains neon acceleration instructions, and partially implements int8 acceleration.|
|TNN_METAL_ENABLE| OFF | Code source/device/metal compilation switch, the code contains metal acceleration instructions.|
|TNN_OPENCL_ENABLE| OFF | Code source/device/opencl compilation switch, the code contains opencl acceleration instructions.|
|TNN_CUDA_ENABLE| OFF | Code source/device/cuda compilation switch, the code contains cuda acceleration instructions, currently only a small part of the implementation has been migrated.|
|TNN_DSP_ENABLE| OFF | Code source/device/dsp compilation switch, currently adapted to snpe implementation.|
|TNN_ATLAS_ENABLE| OFF | The code source/device/atlas compilation switch is currently adapted to Huawei's atlas acceleration framework.|
|TNN_HUAWEI_NPU_ENABLE| OFF | The code source/device/huawei_npu compilation switch is currently adapted to the HiAI acceleration framework.|
|TNN_SYMBOL_HIDE| ON | The symbols of the acceleration library are hidden, and the default non-public interface symbols of release are not visible.|
|TNN_OPENMP_ENABLE| OFF | OpenMP switch, control whether to open openmp acceleration.|
|TNN_BUILD_SHARED| ON | The dynamic library compilation switch, close to compile the static library.|
|TNN_TEST_ENABLE| OFF | test code compilation switch|
|TNN_UNIT_TEST_ENABLE| OFF | Unit test compilation switch, open the unit test compilation switch will automatically turn on the TNN_CPU_ENABLE switch, as a test benchmark.|
|TNN_PROFILER_ENABLE| OFF | Performance debugging switch, after opening it will print more performance information, only for debugging.|
|TNN_QUANTIZATION_ENABLE| OFF | Quantization tool compilation switch|
|TNN_BENCHMARK_MODE| OFF | Benchmark switch, after opening, the model weights file is empty, and data can be automatically generated.|
