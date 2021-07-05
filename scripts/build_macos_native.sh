#!/bin/bash

SHARED_LIB="ON"
METAL="ON"

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

mkdir build_macos_native
cd build_macos_native

cmake ${TNN_ROOT_PATH} \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_X86_ENABLE=ON \
    -DTNN_METAL_ENABLE:BOOL=$METAL \
    -DTNN_UNIT_TEST_ENABLE=ON \
    -DTNN_COVERAGE=ON \
    -DTNN_BENCHMARK_MODE=ON \
    -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB \
    -DTNN_CONVERTER_ENABLE=ON

make -j4
