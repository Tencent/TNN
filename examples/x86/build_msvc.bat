@echo off
SETLOCAL EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
set "TNN_LIB_PATH=!ROOT_DIR!\..\..\scripts\build_msvc\Release\"
set "TNN_OPENVINO_LIB_PATH=!ROOT_DIR!\..\..\source\tnn\network\openvino\thirdparty\openvino\lib"

cd ..\..\scripts
call build_msvc.bat
echo !cd!
cd ..\examples\x86\

rmdir /s /q build_windows
mkdir build_windows
cd build_windows

set VS_FLAG=
set VS_VERSION=
set VSWHERE=

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

cmake -G !VS_FLAG! -A x64 .. ^
    -DCMAKE_SYSTEM_NAME=Windows ^
    -DTNN_LIB_PATH=!TNN_LIB_PATH! ^
    -DTNN_DEMO_WITH_WEBCAM=ON ^
    -DTNN_OPENVINO_LIB_PATH=!TNN_OPENVINO_LIB_PATH!

cmake --build . --config Release -j4

echo "!TNN_LIB_PATH!"
copy "!TNN_LIB_PATH!\TNN.dll" Release\ 
copy "!TNN_LIB_PATH!\..\test\Release\MKLDNNPlugin.dll" Release\
copy "!TNN_LIB_PATH!\..\test\Release\plugins.xml" Release\

:success
    echo Build Successfully!
    goto :eof

:errorHandle
    echo Build Failed !
