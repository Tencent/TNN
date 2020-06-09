#!/bin/bash

TNN_LIB_PATH=$PWD
CC=aarch64-linux-gnu-gcc
CXX=aarch64-linux-gnu-g++

mkdir -p build
cd build
cmake .. \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH

make -j4
