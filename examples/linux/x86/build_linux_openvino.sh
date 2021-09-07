#!/bin/bash

set -euxo pipefail

TNN_LIB_PATH=../../../scripts/build_x86_linux/
TNN_OPENVINO_LIB_PATH=../../../scripts/x86_linux_release/lib/

cd ../../../scripts
sh build_x86_linux.sh
cd -

rm -rf build_linux_openvino
mkdir build_linux_openvino
cd build_linux_openvino

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \
    -DTNN_OPENVINO_ENABLE=ON \
    -DTNN_OPENVINO_LIB_PATH=$TNN_OPENVINO_LIB_PATH \
    -DTNN_DEMO_WITH_OPENCV=OFF \

make -j4
