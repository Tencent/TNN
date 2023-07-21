#!/bin/bash

#CC=aarch64-linux-gnu-gcc
#CXX=aarch64-linux-gnu-g++
TNN_LIB_PATH=../../../scripts/build_atlas/

cd ../../../scripts
./build_atlas.sh
cd -

rm -r build_atlas
mkdir build_atlas
cd build_atlas

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH

make -j4
