@echo off
SETLOCAL EnableDelayedExpansion

set GIT_LFS_SKIP_SMUDGE=1
set OPENVINO_BUILD_SHARED="ON"

set TNN_DIR=%~dp0..\
set BUILD_DIR=%~dp0build_msvc
set OPENVINO_DIR=%BUILD_DIR%\openvino
set TNN_INSTALL_DIR=%~dp0msvc_release

set OPENVINO_INSTALL_DIR=%BUILD_DIR%\openvinoInstallStatic
if %OPENVINO_BUILD_SHARED% == "ON" (
    set OPENVINO_INSTALL_DIR=%BUILD_DIR%\openvinoInstallShared
)
set OPENVINO_ROOT_DIR=%OPENVINO_INSTALL_DIR%

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

call :clone_openvino
if not %return% == 0 (
    goto :errorHandle
)

call :build_openvino
if not %return% == 0 (
    goto :errorHandle
)

echo Building TNN ...
cd %BUILD_DIR%
cmake ^
-DCMAKE_BUILD_TYPE=Release ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DTNN_CPU_ENABLE=ON ^
-DTNN_TEST_ENABLE=ON ^
-DINTTYPES_FORMAT=C99 ^
-DTNN_OPENVINO_ENABLE=ON ^
-DTNN_OPENVINO_BUILD_SHARED=%OPENVINO_BUILD_SHARED% ^
-DTNN_X86_ENABLE=ON ^
../..

cmake --build . --config Release -j4
if !errorlevel! == 1 (
    echo Building Openvino Failed
    goto errorHandle
)

call :pack_tnn

echo "Building Completes. check %TNN_INSTALL_DIR%"

goto :eof

:: Function, pack tnn files
:pack_tnn
    if not exist %TNN_INSTALL_DIR% (
        mkdir %TNN_INSTALL_DIR%
        mkdir %TNN_INSTALL_DIR%\bin
        mkdir %TNN_INSTALL_DIR%\lib
        mkdir %TNN_INSTALL_DIR%\include
    )

    copy %BUILD_DIR%\test\Release\TNNTest.exe %TNN_INSTALL_DIR%\bin\
    copy %BUILD_DIR%\Release\TNN.dll %TNN_INSTALL_DIR%\bin\
    copy %BUILD_DIR%\Release\TNN.lib %TNN_INSTALL_DIR%\lib\

    xcopy /s/e/y %TNN_DIR%\include %TNN_INSTALL_DIR%\include
    copy %OPENVINO_INSTALL_DIR%\deployment_tools\inference_engine\bin\intel64\Release\MKLDNNPlugin.dll %TNN_INSTALL_DIR%\bin\
    copy %OPENVINO_INSTALL_DIR%\deployment_tools\inference_engine\bin\intel64\Release\plugins.xml  %TNN_INSTALL_DIR%\lib\
    copy %OPENVINO_INSTALL_DIR%\deployment_tools\inference_engine\bin\intel64\Release\plugins.xml  %TNN_INSTALL_DIR%\bin\
    copy %OPENVINO_INSTALL_DIR%\deployment_tools\inference_engine\bin\intel64\Release\plugins.xml  %BUILD_DIR%\

    if %OPENVINO_BUILD_SHARED% == "ON" (
        copy %OPENVINO_INSTALL_DIR%\deployment_tools\inference_engine\bin\intel64\Release\inference_engine.dll %TNN_INSTALL_DIR%\bin\
        copy %OPENVINO_INSTALL_DIR%\deployment_tools\inference_engine\bin\intel64\Release\inference_engine_legacy.dll %TNN_INSTALL_DIR%\bin\
        copy %OPENVINO_INSTALL_DIR%\deployment_tools\inference_engine\bin\intel64\Release\inference_engine_transformations.dll %TNN_INSTALL_DIR%\bin\
        copy %OPENVINO_INSTALL_DIR%\deployment_tools\inference_engine\bin\intel64\Release\inference_engine_lp_transformations.dll %TNN_INSTALL_DIR%\bin\
        copy %OPENVINO_INSTALL_DIR%\deployment_tools\ngraph\lib\ngraph.dll %TNN_INSTALL_DIR%\bin\
    )

    goto :returnOk

:: Function, build_openvino
:build_openvino
    if exist %OPENVINO_INSTALL_DIR% (
        goto :returnOk
    )

    if not exist %OPENVINO_DIR%\build (
        mkdir build
    )

    cd %OPENVINO_DIR%/build
    echo Configuring Openvino ...

    cmake ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_SYSTEM_NAME=Windows ^
    -DCMAKE_SYSTEM_PROCESSOR=AMD64 ^
    -DCMAKE_CROSSCOMPILING=OFF ^
    -DENABLE_OPENCV=OFF ^
    -DCMAKE_INSTALL_PREFIX="!OPENVINO_INSTALL_DIR!" ^
    -DENABLE_TBB_RELEASE_ONLY=OFF ^
    -DTHREADING=SEQ ^
    -DNGRAPH_COMPONENT_PREFIX="deployment_tools/ngraph/" ^
    -DNGRAPH_JSON_ENABLE=OFF ^
    -DENABLE_PROFILING_ITT=OFF ^
    -DENABLE_MYRIAD=OFF ^
    -DENABLE_CLDNN=OFF ^
    -DENABLE_GNA=OFF ^
    -DENABLE_VPU=OFF ^
    -DTREAT_WARNING_AS_ERROR=OFF ^
    -DENABLE_SAMPLES=OFF ^
    -DENABLE_SPEECH_DEMO=OFF ^
    -DNGRAPH_ONNX_IMPORT_ENABLE=OFF ^
    ..

    if not !errorlevel! == 0 (
        echo Configure Openvino Failed
        goto :returnError
    )

    cmake --build . --config Release -j4
    if not !errorlevel! == 0 (
        echo Building Openvino Failed
        goto :returnError
    )

    cmake --install .

    goto :returnOk

:: Function, check and clone openvino
:clone_openvino
    cd !BUILD_DIR!
    if not exist !OPENVINO_DIR! (
        git clone --recursive https://github.com/openvinotoolkit/openvino.git
        if !errorlevel! == 1 (
            echo Openvino Clone Failed!
            rd /s /Q openvino
            goto :returnError
        )
    ) 

    cd !OPENVINO_DIR!
    git reset --hard 4795391
    git submodule update
    if !errorlevel! == 1 (
        echo Openvino Clone Failed!
        rd /s /Q openvino
        goto :returnError
    )

    set "sed=!OPENVINO_DIR!\sed\sed.exe"
    if %OPENVINO_BUILD_SHARED% == "OFF" (
        git clone https://github.com/Maosquerade/sed.git
        %sed% -i 152,152s/SHARED/STATIC/g inference-engine/src/inference_engine/CMakeLists.txt
        %sed% -i s/SHARED/STATIC/g inference-engine/src/legacy_api/CMakeLists.txt
        %sed% -i s/SHARED/STATIC/g inference-engine/src/transformations/CMakeLists.txt
        %sed% -i s/SHARED/STATIC/g inference-engine/src/low_precision_transformations/CMakeLists.txt
        %sed% -i s/SHARED/STATIC/g ngraph/src/ngraph/CMakeLists.txt
        %sed% -i "32a\if (CMAKE_SYSTEM_NAME MATCHES \"Windows\")" cmake/developer_package.cmake
        %sed% -i "33a\    set(CMAKE_SYSTEM_PROCESSOR \"AMD64\")" cmake/developer_package.cmake
        %sed% -i "34a\endif()" cmake/developer_package.cmake
        %sed% -i "s/__declspec(dllimport)//g" ngraph/src/ngraph/visibility.hpp
        %sed% -i "s/__declspec(dllimport)//g" inference-engine/include/ie_api.h
    )

    goto :returnOk

:returnOk
    set return=0
    goto :eof

:returnError
    set return=1
    goto :eof

:errorHandle
    echo Building Failed
    goto :eof
