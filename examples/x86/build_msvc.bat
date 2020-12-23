@echo off

set ROOT_DIR=%~dp0
set TNN_LIB_PATH=%ROOT_DIR%\..\..\scripts\msvc_release\lib\
set TNN_BIN_PATH=%ROOT_DIR%\..\..\scripts\msvc_release\bin\
set EXAMPLE_INSTALL_PATH=%ROOT_DIR%\release

cd ..\..\scripts
call build_msvc.bat
echo !cd!
cd ..\examples\x86\

rmdir /s /q build_msvc
mkdir build_msvc
cd build_msvc

cmake -G Ninja .. ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_SYSTEM_NAME=Windows ^
    -DCMAKE_SYSTEM_PROCESSOR=AMD64 ^
    -DTNN_LIB_PATH=%TNN_LIB_PATH% ^
    -DTNN_OPENVINO_LIB_PATH=%OPENVINO_LIB_PATH% ^
    -DTNN_DEMO_WITH_WEBCAM=ON ^
    -DOpenCV_DIR=%OpenCV_DIR%

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

cd %ROOT_DIR%
echo Build Successfully!
goto :eof

:errorHandle
    cd %ROOT_DIR%
    echo Build Failed !

