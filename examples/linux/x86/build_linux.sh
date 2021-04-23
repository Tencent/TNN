#!/bin/bash

set -euxo pipefail

TNN_LIB_PATH=../../../scripts/build_x86_linux/
TNN_OPENVINO_LIB_PATH=../../../scripts/x86_linux_release/lib/

cd ../../../scripts
sh build_x86_linux.sh
cd -

rm -rf build_x86_linux
mkdir build_x86_linux
cd build_x86_linux

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \
    -DTNN_OPENVINO_LIB_PATH=$TNN_OPENVINO_LIB_PATH \
    -DTNN_DEMO_WITH_WEBCAM=OFF \

make -j4
