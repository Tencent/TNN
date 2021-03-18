#!/bin/bash

SHARED_LIB="ON"
ARM="ON"
ARM82="ON"
METAL="OFF"
DEBUG="OFF"
TARGET_ARCH=aarch64

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

rm -rf build_aarch64_macos
mkdir build_aarch64_macos
cd build_aarch64_macos

cmake ${TNN_ROOT_PATH} \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_UNIT_TEST_ENABLE=ON \
    -DDEBUG:BOOL=$DEBUG \
    -DTNN_ARM_ENABLE:BOOL=$ARM \
    -DTNN_ARM82_ENABLE:BOOL=$ARM82 \
    -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
    -DTNN_METAL_ENABLE:BOOL=$METAL \
    -DCMAKE_SYSTEM_PROCESSOR=$TARGET_ARCH \
    -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB


make -j6
