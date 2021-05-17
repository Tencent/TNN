@echo off

set ROOT_DIR=%~dp0
set TNN_LIB_PATH=%ROOT_DIR%\..\..\..\scripts\msvc_release\lib\
set TNN_BIN_PATH=%ROOT_DIR%\..\..\..\scripts\msvc_release\bin\
set EXAMPLE_INSTALL_PATH=%ROOT_DIR%\build_msvc_openvino\release
@REM set OpenCV_DIR=D:\opencv\build\x64\vc15\bin

cd %ROOT_DIR%\..\..\..\scripts\
call build_msvc.bat
echo !cd!
cd %ROOT_DIR%\..\..\..\examples\windows\x86\

:: rmdir /s /q build_msvc_openvino
mkdir build_msvc_openvino
cd build_msvc_openvino

cmake .. ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_SYSTEM_NAME=Windows ^
    -DCMAKE_SYSTEM_PROCESSOR=AMD64 ^
    -DTNN_LIB_PATH=%TNN_LIB_PATH% ^
    -DTNN_OPENVINO_ENABLE=ON ^
    -DTNN_OPENVINO_LIB_PATH=%OPENVINO_LIB_PATH% ^
    -DTNN_DEMO_WITH_OPENCV=OFF
    @REM -DOpenCV_DIR=%OpenCV_DIR%

if !errorlevel! == 1 (
    echo Building TNN Examples Failed
    goto errorHandle
)

cmake --build . --config Release -j4

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
copy %TNN_BIN_PATH%\plugins.xml  %EXAMPLE_INSTALL_PATH%
@REM copy %OpenCV_DIR%\opencv_world452.dll %EXAMPLE_INSTALL_PATH%

cd %ROOT_DIR%
echo Build Successfully!
goto :eof

:errorHandle
    cd %ROOT_DIR%
    echo Build Failed !

