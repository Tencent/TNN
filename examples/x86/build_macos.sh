#!/bin/bash

set -euxo pipefail

TNN_LIB_PATH=../../scripts/build_macos/
TNN_OPENVINO_LIB_PATH=../../source/tnn/network/openvino/thirdparty/openvino/lib

cd ../../scripts
sh build_macos.sh
cd -

rm -rf build_macos
mkdir build_macos
cd build_macos

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \
    -DTNN_OPENVINO_LIB_PATH=$TNN_OPENVINO_LIB_PATH  \
    -DTNN_DEMO_WITH_WEBCAM=ON

make -j4