#!/bin/bash

SHARED_LIB="ON"
ARM="ON"
OPENMP="ON"
OPENCL="OFF"
CC=gcc
CXX=g++
TARGET_ARCH=aarch64

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

mkdir build_aarch64_linux
cd build_aarch64_linux

cmake ${TNN_ROOT_PATH} \
    -DCMAKE_SYSTEM_NAME=Linux  \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DTNN_ARM_ENABLE:BOOL=$ARM \
    -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
    -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_UNIT_TEST_ENABLE=ON \
    -DTNN_COVERAGE=ON \
    -DCMAKE_SYSTEM_PROCESSOR=$TARGET_ARCH \
    -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB


make -j4

# check compile error, or ci will not stop
if [ 0 -ne $? ]
then
    exit -1
fi

ctest --output-on-failure -j 2
