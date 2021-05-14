#!/bin/bash

set -euxo pipefail

TNN_LIB_PATH=../../../scripts/build_macos_native/

cd ../../../scripts
sh build_macos_native.sh
cd -

rm -rf build_macos_native
mkdir build_macos_native
cd build_macos_native

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \
    -DTNN_DEMO_WITH_WEBCAM=OFF

make -j4