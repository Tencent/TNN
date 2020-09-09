
mkdir build_vs2019
cd build_vs2019

cmake -G "Visual Studio 16 2019" -A x64 ^
CMAKE_BUILD_TYPE=RELEASE ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DTNN_NAIVE_ENABLE=ON ^
-DTNN_TEST_ENABLE=ON ^
-DINTTYPES_FORMAT=C99 ^
../..


cmake --build . --config Release
