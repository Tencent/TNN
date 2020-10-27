#!/bin/bash

TNN_LIB_PATH=../../scripts/build_linux/

cd ../../scripts
./build_linux.sh
cd -

rm -r build_linux
mkdir build_linux
cd build_linux
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH

make -j4
