#!/bin/bash

SHARED_LIB="ON"
ARM="ON"
OPENMP="ON"
OPENCL="OFF"
RKNPU="OFF"
CC=aarch64-linux-gnu-gcc
CXX=aarch64-linux-gnu-g++
TARGET_ARCH=aarch64

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

mkdir build_aarch64_linux
cd build_aarch64_linux

cmake ${TNN_ROOT_PATH} \
    -DCMAKE_SYSTEM_NAME=Linux  \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_ARM_ENABLE:BOOL=$ARM \
    -DTNN_TEST_ENABLE:BOOL=ON \
    -DTNN_CPU_ENABLE:BOOL=ON \
    -DTNN_RK_NPU_ENABLE:BOOL=$RKNPU \
    -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
    -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
    -DCMAKE_SYSTEM_PROCESSOR=$TARGET_ARCH \
    -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB


make -j4
