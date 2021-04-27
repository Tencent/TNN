#!/bin/bash

set -euxo pipefail

TNN_LIB_PATH=../../../scripts/build_macos/
TNN_OPENVINO_LIB_PATH=../../../scripts/macos_release/lib/

cd ../../../scripts
sh build_macos.sh
cd -

rm -rf build_macos_openvino
mkdir build_macos_openvino
cd build_macos_openvino

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \
    -DTNN_OPENVINO_ENABLE=ON \
    -DTNN_OPENVINO_LIB_PATH=$TNN_OPENVINO_LIB_PATH  \
    -DTNN_DEMO_WITH_OPENCV=ON

make -j4