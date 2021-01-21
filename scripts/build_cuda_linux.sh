#!/bin/bash

CC=gcc
CXX=g++

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

cd ..

mkdir build_cuda_linux
cd build_cuda_linux

cmake ${TNN_ROOT_PATH} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_CUDA_ENABLE=ON \
    -DTNN_TENSORRT_ENABLE=ON \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DTNN_OPENMP_ENABLE=OFF \
    -DTNN_OPENCL_ENABLE=OFF \
    -DTNN_QUANTIZATION_ENABLE=OFF \
    -DTNN_COVERAGE=OFF \
    -DTNN_BENCHMARK_MODE=OFF \
    -DTNN_BUILD_SHARED=ON \
    -DTNN_CONVERTER_ENABLE=OFF

echo "Building TNN ..."
make -j4

echo "Done"
