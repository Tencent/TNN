#!/bin/bash

set -euxo pipefail

TNN_LIB_PATH=../../../scripts/build_linux_native/

cd ../../../scripts
sh build_linux_native.sh
cd -

rm -rf build_linux_native
mkdir build_linux_native
cd build_linux_native

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH \
    -DTNN_DEMO_WITH_WEBCAM=OFF \

make -j4
