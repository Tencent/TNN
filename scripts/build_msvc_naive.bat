
mkdir build_vs2019_naive
cd build_vs2019_naive

cmake ../.. -G "Ninja" ^ 
CMAKE_BUILD_TYPE=RELEASE ^
-DCMAKE_SYSTEM_NAME=Windows ^
-DTNN_CPU_ENABLE=ON ^
-DTNN_X86_ENABLE=ON ^
-DTNN_TEST_ENABLE=ON ^
-DINTTYPES_FORMAT=C99

cmake --build . --config Release
