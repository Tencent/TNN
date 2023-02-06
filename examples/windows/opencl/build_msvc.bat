@echo off

set ROOT_DIR=%~dp0
set TNN_LIB_PATH=%ROOT_DIR%\..\..\..\scripts\x64_opencl_share_msvc_release\lib\
set TNN_BIN_PATH=%ROOT_DIR%\..\..\..\scripts\x64_opencl_share_msvc_release\bin\
set EXAMPLE_INSTALL_PATH=%ROOT_DIR%\build_x64_opencl\release
set SHARING_MEM_WITH_OPENGL=ON
@REM set OpenCV_DIR=D:\opencv\build\

cd %ROOT_DIR%\..\..\..\scripts\
call build_win_x64_opencl_share.bat
echo !cd!
cd %ROOT_DIR%\..\..\..\examples\windows\opencl\

mkdir build_x64_opencl
cd build_x64_opencl

cmake .. ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_SYSTEM_NAME=Windows ^
    -DCMAKE_SYSTEM_PROCESSOR=AMD64 ^
    -DTNN_LIB_PATH=%TNN_LIB_PATH% ^
    -DTNN_OPENCL_ENABLE=ON ^
    -DSHARING_MEM_WITH_OPENGL=%SHARING_MEM_WITH_OPENGL% ^
    -DTNN_DEMO_WITH_OPENCV=OFF

if !errorlevel! == 1 (
    echo Building TNN Examples Failed
    goto errorHandle
)

cmake --build . --config Release -j8

if !errorlevel! == 1 (
    echo Building TNN Examples Failed
    goto errorHandle
)

if not exist %EXAMPLE_INSTALL_PATH% (
    mkdir %EXAMPLE_INSTALL_PATH%
)

for %%e in (.\*.exe) do copy "%%e" %EXAMPLE_INSTALL_PATH%
for %%e in (%TNN_BIN_PATH%\*.dll) do copy "%%e" %EXAMPLE_INSTALL_PATH%
for /R %OpenCV_DIR% %%e in (*.dll) do copy "%%e" %EXAMPLE_INSTALL_PATH%
@REM copy %OpenCV_DIR%\opencv_world452.dll %EXAMPLE_INSTALL_PATH%

cd %ROOT_DIR%
echo Build Successfully!
goto :eof

:errorHandle
    cd %ROOT_DIR%
    echo Build Failed !