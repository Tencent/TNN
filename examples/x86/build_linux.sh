#!/bin/bash

TNN_LIB_PATH=../../scripts/build_x86/
TNN_OPENVINO_LIB_PATH=../../source/tnn/network/openvino/thirdparty/openvino/lib

cd ../../scripts
sh build_x86_linux.sh
cd -

rm -r build_x86_linux
mkdir build_x86_linux
cd build_x86_linux

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \
    -DTNN_OPENVINO_LIB_PATH=$TNN_OPENVINO_LIB_PATH

make -j4