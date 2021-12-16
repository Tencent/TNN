#!/bin/bash

COMPILER_PATH=${1:-"/usr"}

SHARED_LIB="ON"
ARM="ON"
OPENMP="ON"
OPENCL="OFF"
RKNPU="OFF"
API_LEVEL=18
CC=$COMPILER_PATH/bin/arm-linux-androideabi-gcc
CXX=$COMPILER_PATH/bin/arm-linux-androideabi-g++
TARGET_ARCH=arm

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

rm -rf build_androideabi_linux
mkdir build_androideabi_linux
cd build_androideabi_linux

cmake ${TNN_ROOT_PATH} \
    -DCMAKE_SYSTEM_NAME=Linux  \
    -DANDROID_API_LEVAL=$API_LEVEL \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DDEBUG=OFF \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_ARM_ENABLE:BOOL=$ARM \
    -DTNN_RK_NPU_ENABLE:BOOL=$RKNPU \
    -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
    -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
    -DCMAKE_SYSTEM_PROCESSOR=$TARGET_ARCH \
    -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB

make -j7
