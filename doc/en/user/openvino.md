# TNN X86/Openvino Documentation

## Compile with scripts
### 1. Use scripts/build_openvino.sh (or scripts/build_openvino.bat for Windows system) to compile Openvino and TNN
```
$ cd scripts/
$ sh build_openvino.sh
```

### 2. Complile manually
#### Linux
Recommended cloning from [Openvino github](https://github.com/openvinotoolkit/openvino) (on commit 9df6a8f). Modify the CMakeLists for building with static version. For details move to [scripts/build_openvino.sh](../../../scripts/build_openvino.sh)
With building Openvino successfully, add Openvino lib and include file to source tnn directory:
```
source/tnn/network/openvino/thirdparty/openvino/lib:
libinference_engine.a
libinference_engine_legacy.a
libinference_engine_transformations.a
libinference_engine_lp_transformations.a
libMKLDNNPlugin.so
libngraph.a
libpugixml.a

source/tnn/network/openvino/thirdparty/openvino/:
openvino_install_path/deployment_tools/inference_engine/include

source/tnn/network/openvino/thirdparty/ngraph/:
openvino_install_path/deployment_tools/ngraph/include

TNN build directory:
plugins.xml
```
Then build TNN with following command:
```
cmake .. \
    -DTNN_OPENIVNO_ENABLE=ON \
    -DTNN_X86_ENABLE=ON \
    -DTNN_TEST_ENABLE=ON 

make -j4
```

#### Windows
Enviroment Requests: Visual Studio 2019 <br>
Recommended cloning from [Openvino github](https://github.com/openvinotoolkit/openvino) (on commit 9df6a8f). Modify the CMakeLists for building with static version. For details move to [scripts/build_openvino.bat](../../../scripts/build_openvino.bat)
With building Openvino successfully, add Openvino lib and include file to source tnn directory:
```
source/tnn/network/openvino/thirdparty/openvino/lib:
inference_engine.lib
inference_engine_legacy.lib
inference_engine_transformations.lib
inference_engine_lp_transformations.lib
MKLDNNPlugin.lib
ngraph.lib
pugixml.lib

source/tnn/network/openvino/thirdparty/openvino/:
openvino_install_path/deployment_tools/inference_engine/include

source/tnn/network/openvino/thirdparty/ngraph:
openvino_install_path/deployment_tools/ngraph/include

files needed when running:
plugins.xml MKLDNNPlugin.dll
```
Then build TNN with following commad:
```
cmake .. ^
    -G "Visual Studio 16 2019" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_SYSTEM_NAME=Windows ^
    -DTNN_TEST_ENABLE=ON ^
    -DINTTYPES_FORMAT=C99 ^
    -DTNN_OPENVINO_ENABLE=ON ^
    -DTNN_X86_ENABLE=ON ^

cmake --build . --config Release
```

## How to run
### 1. Run with intergrated test file
Move to ```build_openvino/test/```, run ```TNNTest``` with model, and set ```network_type=OPENVINO```(required)
```
$ cd build_openvino/test/
$ ./TNNTest -mp PATH_TO_MODEL -nt OPENVINO -ip PATH_TO_INPUT -op PATH_TO_OUTPUT
```

### 2. API Documentation
Refer to [API Documentation](doc/en/user/api.md), which needs to set ```config.device_type``` as ```DEVICE_X86``` and ```config.network_type``` as ```NETWORK_TYPE_OPENVINO```
```cpp
config.device_type  = TNN_NS::DEVICE_X86
config.network_type = TNN_NS::NETWORK_TYPE_OPENVINO
```

## Run with demo
Move to ```example/openivno/``` and run ```build_openvino.sh``` to compile demos with x86 architecture. Then call ```demo_x86_linux_imageclassify``` or ```demo_x86_linux_facedetector``` to run demos. For details move to 
