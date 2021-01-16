#!/bin/bash

SHARED_LIB="ON"
ARM="ON"
OPENMP="ON"
OPENCL="OFF"
CC=arm-linux-gnueabihf-gcc
CXX=arm-linux-gnueabihf-g++
TARGET_ARCH=arm

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

mkdir build_armhf_linux
cd build_armhf_linux

cmake ${TNN_ROOT_PATH} \
    -DCMAKE_SYSTEM_NAME=Linux  \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=OFF \
    -DDEBUG=OFF \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_ARM_ENABLE:BOOL=$ARM \
    -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
    -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
    -DCMAKE_SYSTEM_PROCESSOR=$TARGET_ARCH \
    -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB

make -j4
