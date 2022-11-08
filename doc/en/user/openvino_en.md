# TNN X86/Openvino Documentation
## Introduction to TNN X86/Openvino
This module supports TNN on x86 architecture and includes Openvino framework into TNN, which allows a TNN model running on Openvino Network.

## Environment Requirements
### Linux
Cmake(>=3.7.2)
### Windows
Visual Stuido(>=2017)<br>
CMake(>=3.7.2 or use build-in CMake in Visual Studio Tools)

## Compile with scripts
```
Linux:
$ cd scripts/
$ sh build_linux.sh

Windows:
cd scripts\
.\build_msvc.bat [VS2015/VS2017/VS2019]
```
Refer to [FAQ](#FAQ) if failed.


## How to run
### 1. Run with intergrated test file
Move to ```build_openvino/test/```, run ```TNNTest``` with model, and set device_type to X86
```
$ cd build_openvino/test/
$ ./TNNTest -mp PATH_TO_MODEL -dt X86 -ip PATH_TO_INPUT -op PATH_TO_OUTPUT
```

### 2. API Documentation
Refer to [API Documentation](api_en.md), which needs to set ```config.device_type``` as ```DEVICE_X86``` and ```config.network_type``` as ```NETWORK_TYPE_OPENVINO```
```cpp
config.device_type  = TNN_NS::DEVICE_X86
// run with native x86 optimized code, if network type is not set
config.network_type = TNN_NS::NETWORK_TYPE_OPENVINO
```

## Run with demo
Move to ```example/openivno/``` and run ```build_openvino.sh``` to compile demos with x86 architecture. Then call ```demo_x86_linux_imageclassify``` or ```demo_x86_linux_facedetector``` to run demos. For details move to 

## Run Demo
Refer to [demo documentaion](demo_en.md)

## FAQ
Q: CMake not found in Windows?<br>
A: If CMake was installed, add the CMake path to Windows Environment Viraibles. Or use Visual Studio Prompt to run build_x86_msvc.bat, which includes build-in CMake.

Q: Visual Studio not found in Windows?<br>
A: Execute the scripts with Visual Studio Version, Like
```
.\build_x86_msvc.bat VS2019
```

Q: Error 0x4001 or 16385 with message "Invalid Model Content"<br>
A: set `std::ios::binary` when reading Model stream:
```cpp
std::ifstream model_stream(mode_path, std::ios::binary);
```