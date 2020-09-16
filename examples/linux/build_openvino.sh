#!/bin/bash

TNN_LIB_PATH=../../scripts/build_openvino/
OPENVINO_LIB_PATH=../../source/tnn/network/openvino/thirdparty/openvino/lib

cd ../../scripts
./build_openvino.sh
cd -

rm -r build_openvino
mkdir build_openvino
cd build_openvino
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \
    -DTNN_OPENVINO_ENABLE=ON \
    -DTNN_OPENVINO_LIB_PATH=$OPENVINO_LIB_PATH

make -j4