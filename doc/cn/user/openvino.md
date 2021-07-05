# TNN X86/Openvino 使用文档
## TNN X86/Openvino 介绍
本模块支持 x86 架构，封装了 OPENVINO 的框架到 TNN 内部，允许使用 TNN 的模型跑 OPENVINO 的网络。

## 环境要求
### Linux
CMake(>=3.7.2)
### Windows
Visual Studio(>=2017) <br>
CMake(>=3.7.2) 或使用 Visual Studio 的 CMake 工具
## 编译方法
    
使用脚本快速部署：
```
Linux:
$ cd scripts/
$ sh build_linux.sh

Windows:
cd scripts\
.\build_msvc.bat
```  
如编译失败，请参考[常见问题](#常见问题)


<!-- ### 2. 手动编译
#### Linux
建议从 github 代码上手动编译安装 openvino (commit 9df6a8f)，并修改文件将其编译成静态版本，具体编译方法可参考脚本 scripts/build_openvino.sh
安装完成后参照编译脚本将 ```inference_engine``` 和 ```ngraph``` 的 ```include``` 及 ```lib``` 文件放入 ```source/tnn/network/openvino/thirdparty``` 目录，具体要求的文件及目录如下：<br/>
```
source/tnn/network/openvino/thirdparty/openvino/lib 需要的文件
libinference_engine.a
libinference_engine_legacy.a
libinference_engine_transformations.a
libinference_engine_lp_transformations.a
libMKLDNNPlugin.so
libngraph.a
libpugixml.a

source/tnn/network/openvino/thirdparty/openvino/ 需要的文件
openvino_install_path/deployment_tools/inference_engine/include

source/tnn/network/openvino/thirdparty/ngraph 需要的文件
openvino_install_path/deployment_tools/ngraph/include

TNN build 目录下需要的文件
plugins.xml
```
文件放置完成后，使用如下命令编译
```
cmake .. \
    -DTNN_OPENIVNO_ENABLE=ON \
    -DTNN_X86_ENABLE=ON \
    -DTNN_TEST_ENABLE=ON 

make -j4
```

#### Windows
环境要求：Visual Studio 开发环境（VS2019）<br>
建议从 github 代码上手动编译安装 openvino (commit 9df6a8f)，并修改文件将其编译成静态版本，具体编译方法可参考脚本 scripts/build_openvino.bat
安装完成后参照编译脚本将 ```inference_engine``` 和 ```ngraph``` 的 ```include``` 及 ```lib``` 文件放入 ```source/tnn/network/openvino/thirdparty``` 目录，具体要求的文件及目录如下：<br/>
```
source/tnn/network/openvino/thirdparty/openvino/lib 需要的文件
inference_engine.lib
inference_engine_legacy.lib
inference_engine_transformations.lib
inference_engine_lp_transformations.lib
MKLDNNPlugin.lib
ngraph.lib
pugixml.lib

source/tnn/network/openvino/thirdparty/openvino/ 需要的文件
openvino_install_path/deployment_tools/inference_engine/include

source/tnn/network/openvino/thirdparty/ngraph 需要的文件
openvino_install_path/deployment_tools/ngraph/include

运行需要的文件
plugins.xml MKLDNNPlugin.dll
```
文件放置完成后，使用如下命令编译
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
``` -->

## 使用方法
### 1.  快速运行
进入 build_x86/test/ 目录，使用 TNNTest 运行模型，并指定 ```-dt X86```
```
$ cd build_openvino/test/
$ ./TNNTest -mp PATH_TO_MODEL -dt X86 -ip PATH_TO_INPUT -op PATH_TO_OUTPUT
```
### 2.  API 调用
参考 [API 调用](api.md)，需要在初始化网络时设置 config.device_type 为 DEVICE_X86，config.network_type 为 NETWORK_TYPE_OPENVINO
```cpp
config.device_type  = TNN_NS::DEVICE_X86
// 如果network_type不设的话，则运行的是原生的X86优化
config.network_type = TNN_NS::NETWORK_TYPE_OPENVINO
```

## demo 运行
参考 [demo 文档](demo.md)

## 常见问题
Q: Windows 找不到 Cmake？ <br>
A: 如果本机安装了 Cmake，将环境变量添加到 Path 中，或者使用 Visual Studio Prompt 运行脚本 build_x86_msvc.bat。

Q: Windows 找不到 Visual Studio？
A: 运行脚本时加上自己的 VS 版本：如 ```.\build_x86_msvc.bat VS2019```

Q：git 克隆 Openvino 仓库失败 <br>
A：配置 git 代理：
```
git config --global https.proxy http://127.0.0.1:1080
git config --global https.proxy https://127.0.0.1:1080
```

Q: 运行时报错 0x4001 或 16385，报错信息 Invalid Model Content<br>
A: 读入 model 流时设置为 ```std::ios::binary```
```cpp
std::ifstream model_stream(model_path, std::ios::binary);
```