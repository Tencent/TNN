echo off
set TNN_DIR=%~dp0\..\
set thirdparty_dir=%TNN_DIR%\source\tnn\network\openvino\thirdparty

if not exist %thirdparty_dir% (
    mkdir %thirdparty_dir%\openvino
    mkdir %thirdparty_dir%\openvino\lib
    mkdir %thirdparty_dir%\ngraph
)
echo on

if not exist %TNN_DIR%\scripts\build_openvino\ (
    mkdir build_openvino
)
cd build_openvino

if not exist %TNN_DIR%\scripts\build_openvino\openvino (
    git clone https://github.com/openvinotoolkit/openvino.git
)

echo off
set OPENVINO_DIR=%TNN_DIR%\scripts\build_openvino\openvino
echo on

cd openvino
git reset --hard 9df6a8f
git submodule update --init --recursive

echo off
set "rar=C:\Program Files\WinRAR\WinRAR.exe"
if not exist ./sed (
    bitsadmin /transfer "Myjob" http://unxutils.sourceforge.net/UnxUpdates.zip %OPENVINO_DIR%\UnxUpdates.zip
    "%rar%" x -ad -y *.zip ./
    del *.zip
)

set sed="UnxUpdates\sed.exe"
%sed% -i 152,152s/SHARED/STATIC/g inference-engine/src/inference_engine/CMakeLists.txt
%sed% -i s/SHARED/STATIC/g inference-engine/src/legacy_api/CMakeLists.txt
%sed% -i s/SHARED/STATIC/g inference-engine/src/transformations/CMakeLists.txt
%sed% -i s/SHARED/STATIC/g inference-engine/src/low_precision_transformations/CMakeLists.txt
%sed% -i s/SHARED/STATIC/g ngraph/src/ngraph/CMakeLists.txt
%sed% -i "32a\if (CMAKE_SYSTEM_NAME MATCHES \"Windows\")" cmake/developer_package.cmake
%sed% -i "33a\    set(CMAKE_SYSTEM_PROCESSOR \"AMD64\")" cmake/developer_package.cmake
%sed% -i "34a\endif()" cmake/developer_package.cmake
%sed% -i s/__declspec(dllimport)//g ngraph/src/ngraph/visibility.hpp
%sed% -i s/__declspec(dllimport)//g inference-engine/include/ie_api.h
echo on

if not exist %OPENVINO_DIR%\build (
    mkdir build
)
cd build

cmake -G "Visual Studio 16 2019" -A x64 ^
-DCMAKE_BUILD_TYPE=Release ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DCMAKE_SYSTEM_PROCESSOR=AMD64 ^
-DCMAKE_CROSSCOMPILING=OFF ^
-DENABLE_OPENCV=OFF ^
-DCMAKE_INSTALL_PREFIX="%TNN_DIR%\scripts\build_openvino\openvinoInstall" ^
-DENABLE_CLDNN=OFF ^
-DENABLE_TBB_RELEASE_ONLY=OFF ^
-DTHREADING=SEQ ^
-DNGRAPH_COMPONENT_PREFIX="deployment_tools/ngraph/" ^
-DENABLE_MYRIAD=OFF ^
-DNGRAPH_JSON_ENABLE=OFF ^
-DENABLE_PROFILING_ITT=OFF ^
-DENABLE_GNA=OFF ^
-DENABLE_VPU=OFF ^
-DTREAT_WARNING_AS_ERROR=OFF ^
-DENABLE_SAMPLES=OFF ^
-DENABLE_SPEECH_DEMO=OFF ^
-DNGRAPH_ONNX_IMPORT_ENABLE=OFF ^
..

cmake --build . --config Release -j4
cmake --install .

cd ../..

cd build_openvino

echo off
xcopy /s/e/y "%TNN_DIR%\scripts\build_openvino\openvinoInstall\deployment_tools\inference_engine\include" "%thirdparty_dir%\openvino\include\"
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\inference_engine.lib %thirdparty_dir%\openvino\lib\
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\inference_engine_legacy.lib %thirdparty_dir%\openvino\lib\
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\inference_engine_transformations.lib %thirdparty_dir%\openvino\lib\
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\inference_engine_lp_transformations.lib %thirdparty_dir%\openvino\lib\
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\MKLDNNPlugin.lib %thirdparty_dir%\openvino\lib\

xcopy /s/e/y "%TNN_DIR%\scripts\build_openvino\openvinoInstall\deployment_tools\ngraph\include" "%thirdparty_dir%\ngraph\include\"
copy openvinoInstall\deployment_tools\ngraph\lib\ngraph.lib %thirdparty_dir%\openvino\lib
copy openvinoInstall\lib\pugixml.lib  %thirdparty_dir%\openvino\lib
echo on

cmake -G "Visual Studio 16 2019" -A x64 ^
-DCMAKE_BUILD_TYPE=Release ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DTNN_CPU_ENABLE=ON ^
-DTNN_TEST_ENABLE=ON ^
-DINTTYPES_FORMAT=C99 ^
-DTNN_OPENVINO_ENABLE=ON ^
-DDEBUG=ON ^
../..

cmake --build . --config Release -j4

echo off
copy Release\TNN.dll test\Release\TNN.dll
copy openvinoInstall\deployment_tools\inference_engine\bin\intel64\Release\plugins.xml test\Release\plugins.xml
copy openvinoInstall\deployment_tools\inference_engine\bin\intel64\Release\MKLDNNPlugin.dll test\Release\MKLDNNPlugin.dll