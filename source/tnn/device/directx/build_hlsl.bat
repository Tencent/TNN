:: 递归搜索文件夹下的所有hlsl 文件并编译
:: %1 hlsl 文件所在的文件夹

@echo off

set CUR_DIR=%~dp0
set TNN_HLSL_PATH=%1
set COMPILE_FLAGS=/E CSMain /T cs_5_0 /Fh

cd %1

for %%f in (*.hlsl) do (
    if "%%~xf"==".hlsl" (
      echo %%f
      fxc %%f %COMPILE_FLAGS% /Vn g_%%~nf
    )
)

cd %CUR_DIR%