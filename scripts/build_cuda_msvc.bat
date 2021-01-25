@echo on
SETLOCAL EnableDelayedExpansion

if defined CUDA_PATH (
    rem set CUDACXX=%CUDA_PATH%\bin\nvcc.exe
)else (
    echo "CUDA Compiler not found. try install it and set the CUDA_PATH environment variable"
    goto :eof
)

set TNN_DIR=%~dp0..\
set BUILD_DIR=%~dp0build_cuda_msvc
set TENSORRT_ROOT_DIR=E:\Deps\TensorRT-7.0.0.11
set CUDNN_ROOT_DIR=E:\Deps\cudnn-7.6.5
set TNN_INSTALL_DIR=%~dp0cuda_msvc_release

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

echo Building TNN ...
cd %BUILD_DIR%
cmake -G Ninja ^
-DCMAKE_BUILD_TYPE=Release ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DTNN_CUDA_ENABLE=ON ^
-DTNN_TENSORRT_ENABLE=ON ^
-DTNN_TEST_ENABLE=ON ^
-DINTTYPES_FORMAT=C99 ^
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

    :: include
    xcopy /s/e/y %TNN_DIR%\include %TNN_INSTALL_DIR%\include

    :: lib
    copy %BUILD_DIR%\TNN.lib %TNN_INSTALL_DIR%\lib\

    :: bin
    copy %BUILD_DIR%\TNN.dll %TNN_INSTALL_DIR%\bin\
    copy %BUILD_DIR%\test\TNNTest.exe %TNN_INSTALL_DIR%\bin\

    :: deps bin
    copy %TENSORRT_ROOT_DIR%\lib\nvinfer.dll %TNN_INSTALL_DIR%\bin\
    copy %TENSORRT_ROOT_DIR%\lib\nvinfer_plugin.dll %TNN_INSTALL_DIR%\bin\
    copy %CUDNN_ROOT_DIR%\bin\cudnn64_7.dll %TNN_INSTALL_DIR%\bin\

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
