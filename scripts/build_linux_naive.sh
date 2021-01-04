#!/bin/bash

SHARED_LIB="ON"
ARM="ON"
OPENMP="ON"
OPENCL="OFF"
QUANTIZATION="OFF"
CC=gcc
CXX=g++

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

mkdir build_linux_naive
cd build_linux_naive

cmake ${TNN_ROOT_PATH} \
    -DCMAKE_SYSTEM_NAME=Linux  \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DTNN_ARM_ENABLE:BOOL=$ARM \
    -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
    -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
    -DTNN_QUANTIZATION_ENABLE:BOOL=$QUANTIZATION \
    -DTNN_UNIT_TEST_ENABLE=ON \
    -DTNN_COVERAGE=ON \
    -DTNN_BENCHMARK_MODE=ON \
    -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB \
    -DTNN_CONVERTER_ENABLE=ON


make -j4

# check compile error, or ci will not stop
if [ 0 -ne $? ]
then
    exit -1
fi

ctest --output-on-failure -j 2
