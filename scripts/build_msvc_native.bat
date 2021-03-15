set TNN_DIR=%~dp0..\

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
        call "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars32.bat"
    ) else (
        echo "build x64"
        call "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
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
-DTNN_TEST_ENABLE=ON ^
-DINTTYPES_FORMAT=C99

cmake --build . --config Release
