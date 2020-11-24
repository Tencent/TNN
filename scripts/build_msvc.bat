@echo off
SETLOCAL EnableDelayedExpansion

set TNN_DIR=%~dp0\..\
set thirdparty_dir=!TNN_DIR!\source\tnn\network\openvino\thirdparty

REM 确认 cmake 可用
set CMAKE_PATH=
for /f "usebackq tokens=*" %%i in (`where cmake`) do (
    set "CMAKE_PATH=%%i"
)
if "!CMAKE_PATH!"=="" (
    echo CMake not found
    goto errorHandle
)

REM 确认 Visual Studio 版本
set ROOT_DIR="%~dp0"
set VS_FLAG=
set VS_VERSION=
set VSWHERE=
set GIT_LFS_SKIP_SMUDGE=1

if not "%1" == "" (
    if "%1"=="VS2015" (
        set "VS_VERSION=2015"
    ) else if "%1" == "VS2017" (
        set "VS_VERSION=2017"
    ) else if "%1" == "VS2019" (
        set "VS_VERSION=2019"
    )
) else (
    if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
        set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    ) else if exist "%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" (
        set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
    ) else (
        echo "Visual Studio not found"
        goto errorHandle
    )
    echo Searching Visual Studio !VS_VERSION!...
    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -products * -requires Microsoft.Component.MSBuild -property catalog_productLineVersion`) do (
        set "VS_VERSION=%%i"
    )
)

if "!VS_VERSION!"=="2019" (
    set "VS_FLAG="Visual Studio 16 2019""
) else if "!VS_VERSION!"=="2017" (
    set "VS_FLAG="Visual Studio 15 2017""
) else if "!VS_VERSION!"=="2015" (
    set "VS_FLAG="Visual Studio 14 2015""
) else (
    echo "Visual Studio version too low, require VS2015 at least"
    goto errorHandle
)

echo Making thirdparty directory ...
if not exist !thirdparty_dir! (
    mkdir !thirdparty_dir!\openvino
    mkdir !thirdparty_dir!\openvino\lib
    mkdir !thirdparty_dir!\ngraph
)

if not exist !TNN_DIR!\scripts\build_msvc\ (
    mkdir build_msvc
)
cd build_msvc

if not exist !TNN_DIR!\scripts\build_msvc\openvino (
    git clone https://github.com/openvinotoolkit/openvino.git
) else (
    goto buildTNN
)

if !errorlevel! == 1 (
    echo Openvino Clone Failed!
    rd /s /Q openvino
    goto errorHandle
)

set OPENVINO_DIR=!TNN_DIR!\scripts\build_msvc\openvino

cd openvino
git reset --hard 9df6a8f
git submodule update --init --recursive

git clone https://github.com/Maosquerade/sed.git
set "sed=!TNN_DIR!\scripts\build_msvc\openvino\sed\sed.exe"
echo !sed!
!sed! -i 152,152s/SHARED/STATIC/g inference-engine/src/inference_engine/CMakeLists.txt
!sed! -i s/SHARED/STATIC/g inference-engine/src/legacy_api/CMakeLists.txt
!sed! -i s/SHARED/STATIC/g inference-engine/src/transformations/CMakeLists.txt
!sed! -i s/SHARED/STATIC/g inference-engine/src/low_precision_transformations/CMakeLists.txt
!sed! -i s/SHARED/STATIC/g ngraph/src/ngraph/CMakeLists.txt
!sed! -i "32a\if (CMAKE_SYSTEM_NAME MATCHES \"Windows\")" cmake/developer_package.cmake
!sed! -i "33a\    set(CMAKE_SYSTEM_PROCESSOR \"AMD64\")" cmake/developer_package.cmake
!sed! -i "34a\endif()" cmake/developer_package.cmake
!sed! -i s/__declspec(dllimport)//g ngraph/src/ngraph/visibility.hpp
!sed! -i s/__declspec(dllimport)//g inference-engine/include/ie_api.h

if not exist %OPENVINO_DIR%\build (
    mkdir build
)
cd build

echo Configuring Openvino ...

cmake -G !VS_FLAG! -A x64 ^
-DCMAKE_BUILD_TYPE=Release ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DCMAKE_SYSTEM_PROCESSOR=AMD64 ^
-DCMAKE_CROSSCOMPILING=OFF ^
-DENABLE_OPENCV=OFF ^
-DCMAKE_INSTALL_PREFIX="!TNN_DIR!\scripts\build_msvc\openvinoInstall" ^
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

if !errorlevel! == 1 (
    echo Configure Openvino Failed
    goto errorHandle
)

cmake --build . --config Release -j4
if !errorlevel! == 1 (
    echo Building Openvino Failed
    goto errorHandle
)

cmake --install .

cd ../..

cd build_msvc

xcopy /s/e/y "!TNN_DIR!\scripts\build_msvc\openvinoInstall\deployment_tools\inference_engine\include" "!thirdparty_dir!\openvino\include\"
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\inference_engine.lib !thirdparty_dir!\openvino\lib\
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\inference_engine_legacy.lib !thirdparty_dir!\openvino\lib\
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\inference_engine_transformations.lib !thirdparty_dir!\openvino\lib\
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\inference_engine_lp_transformations.lib !thirdparty_dir!\openvino\lib\
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\MKLDNNPlugin.lib !thirdparty_dir!\openvino\lib\
copy openvinoInstall\deployment_tools\inference_engine\lib\intel64\Release\plugins.xml !thirdparty_dir!\openvino\lib\

xcopy /s/e/y "!TNN_DIR!\scripts\build_msvc\openvinoInstall\deployment_tools\ngraph\include" "!thirdparty_dir!\ngraph\include\"
copy openvinoInstall\deployment_tools\ngraph\lib\ngraph.lib !thirdparty_dir!\openvino\lib
copy openvinoInstall\lib\pugixml.lib  !thirdparty_dir!\openvino\lib

:buildTNN
echo Building TNN ...
cmake -G !VS_FLAG! -A x64 ^
-DCMAKE_BUILD_TYPE=Release ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DTNN_CPU_ENABLE=ON ^
-DTNN_TEST_ENABLE=ON ^
-DINTTYPES_FORMAT=C99 ^
-DTNN_OPENVINO_ENABLE=ON ^
-DTNN_X86_ENABLE=ON ^
../..

cmake --build . --config Release -j4
if !errorlevel! == 1 (
    echo Building Openvino Failed
    goto errorHandle
)

copy Release\TNN.dll test\Release\TNN.dll
copy openvinoInstall\deployment_tools\inference_engine\bin\intel64\Release\plugins.xml test\Release\plugins.xml
copy openvinoInstall\deployment_tools\inference_engine\bin\intel64\Release\MKLDNNPlugin.dll test\Release\MKLDNNPlugin.dll

echo build Successfully
goto :eof

:errorHandle
    echo Building Failed