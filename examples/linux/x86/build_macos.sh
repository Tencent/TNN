#!/bin/bash

set -euxo pipefail

TNN_LIB_PATH=../../../scripts/build_macos/
TNN_OPENVINO_LIB_PATH=../../../scripts/macos_release/lib/

cd ../../../scripts
sh build_macos.sh
cd -

rm -rf build_x86_macos
mkdir build_x86_macos
cd build_x86_macos

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \
    -DTNN_OPENVINO_LIB_PATH=$TNN_OPENVINO_LIB_PATH  \
    -DTNN_DEMO_WITH_OPENCV=ON

make -j4