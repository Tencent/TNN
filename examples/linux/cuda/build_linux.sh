#!/bin/bash

TNN_LIB_PATH=../../../scripts/build_cuda_linux/

cd ../../../scripts
sh build_cuda_linux.sh
cd -

rm -rf build_cuda_linux
mkdir build_cuda_linux
cd build_cuda_linux

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \

make -j4
