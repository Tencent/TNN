@echo off
SETLOCAL EnableDelayedExpansion

set TNN_DIR=%~dp0..\
set BUILD_DIR=%~dp0build_x64_opencl_share
set TNN_INSTALL_DIR=%~dp0x64_opencl_share_msvc_release
set BUILD_TYPE=Release
set TNN_OPENCL_PREFER_GPU_TYPE=INTEL_GPU

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

echo Building TNN ...
cd %BUILD_DIR%
cmake -G "Visual Studio 16 2019" ^
-DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DCMAKE_SYSTEM_VERSION="10.0" ^
-DTNN_OPENCL_ENABLE=ON ^
-DTNN_X86_ENABLE=ON ^
-DTNN_TEST_ENABLE=ON ^
-DTNN_DYNAMIC_RANGE_QUANTIZATION_ENABLE=OFF ^
-DINTTYPES_FORMAT=C99 ^
-DSHARING_MEM_WITH_OPENGL=ON ^
-DTNN_OPENCL_PREFER_GPU_TYPE=%TNN_OPENCL_PREFER_GPU_TYPE% ^
../..

cmake --build . --config %BUILD_TYPE% -j8
if !errorlevel! == 1 (
    echo Building TNN Failed
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
    copy %BUILD_DIR%\%BUILD_TYPE%\TNN.lib %TNN_INSTALL_DIR%\lib\

    :: bin
    copy %BUILD_DIR%\%BUILD_TYPE%\TNN.dll %TNN_INSTALL_DIR%\bin\
    copy %BUILD_DIR%\test\%BUILD_TYPE%\TNNTest.exe %TNN_INSTALL_DIR%\bin\

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

