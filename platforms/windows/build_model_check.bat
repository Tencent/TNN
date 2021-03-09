set TNN_DIR=%~dp0..\..\

@echo off
echo %TNN_DIR%
echo %1

if "%2" == "" (
    goto init_fold
) else (
    goto init_env
)

:init_env
    if %1 == x86 (
        echo "build x86"
        call "D:\Microsoft\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars32.bat"
    ) else (
        echo "build x64"
        call "D:\Microsoft\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
    )
    goto init_fold

:init_fold
    mkdir build_win
    cd build_win

cmake %TNN_DIR% -G "Ninja" ^
-DCMAKE_BUILD_TYPE=Release ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DTNN_CPU_ENABLE=ON ^
-DTNN_X86_ENABLE=ON ^
-DTNN_OPENCL_ENABLE=ON ^
-DTNN_MODEL_CHECK_ENABLE:BOOL="ON" ^
-DTNN_BUILD_SHARED:BOOL="ON" ^
-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE ^
-DINTTYPES_FORMAT=C99

cmake --build . --config Release