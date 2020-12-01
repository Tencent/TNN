#!/bin/bash

set -euxo pipefail

TNN_LIB_PATH=../../scripts/build_linux/
TNN_OPENVINO_LIB_PATH=../../source/tnn/network/openvino/thirdparty/openvino/lib

cd ../../scripts
sh build_linux.sh
cd -

rm -rf build_linux
mkdir build_linux
cd build_linux

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \
    -DTNN_OPENVINO_LIB_PATH=$TNN_OPENVINO_LIB_PATH \
    -DTNN_DEMO_WITH_WEBCAM=ON \

make -j4