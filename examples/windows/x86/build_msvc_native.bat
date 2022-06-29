@echo off

set ROOT_DIR=%~dp0
set TNN_LIB_PATH=%ROOT_DIR%\..\..\..\scripts\build_win\
set TNN_BIN_PATH=%ROOT_DIR%\..\..\..\scripts\build_win\
set EXAMPLE_INSTALL_PATH=%ROOT_DIR%\build_msvc_native\release

cd %ROOT_DIR%\..\..\..\scripts\
call build_msvc_native.bat
echo !cd!
cd %ROOT_DIR%\..\..\..\examples\windows\x86\

:: rmdir /s /q build_msvc_native
mkdir build_msvc_native
cd build_msvc_native

cmake .. ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_SYSTEM_NAME=Windows ^
    -DCMAKE_SYSTEM_PROCESSOR=AMD64 ^
    -DTNN_LIB_PATH=%TNN_LIB_PATH% ^
    -DTNN_DEMO_WITH_WEBCAM=OFF
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

cd %ROOT_DIR%
echo Build Successfully!
goto :eof

:errorHandle
    cd %ROOT_DIR%
    echo Build Failed !

