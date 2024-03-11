#!/bin/bash

DEBUG=0
TARGET_ARCH=aarch64
TORCHAIE="ON"

export LIBTORCH_ROOT_DIR=/usr/local/python3.9.2/lib/python3.9/site-packages/torch
export TORCHAIE_ROOT_DIR=/usr/local/Ascend/torch_aie/latest

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

TNN_BUILD_DIR=${TNN_ROOT_PATH}/scripts/build_atlas
TNN_INSTALL_DIR=${TNN_ROOT_PATH}/scripts/release_atlas
if [ $DEBUG == "ON" ]; then
    TNN_BUILD_DIR=${TNN_ROOT_PATH}/scripts/build_atlas_debug
    TNN_INSTALL_DIR=${TNN_ROOT_PATH}/scripts/release_atlas_debug
fi

TNN_VERSION_PATH=$TNN_ROOT_PATH/scripts/version
cd $TNN_VERSION_PATH
source $TNN_VERSION_PATH/version.sh
source $TNN_VERSION_PATH/add_version_attr.sh

mkdir -p ${TNN_BUILD_DIR}
cd ${TNN_BUILD_DIR}


cmake ${TNN_ROOT_PATH} \
      -DCMAKE_BUILD_TYPE=Release \
      -DDEBUG=$DEBUG \
      -DTNN_TEST_ENABLE:BOOL="ON" \
      -DTNN_BENCHMARK_MODE:BOOL="OFF" \
      -DTNN_CPU_ENABLE:BOOL="ON"  \
      -DTNN_ARM_ENABLE:BOOL="ON" \
      -DTNN_OPENMP_ENABLE:BOOL="ON" \
      -DTNN_X86_ENABLE:BOOL="OFF"  \
      -DTNN_BUILD_SHARED:BOOL="ON" \
      -DCMAKE_SYSTEM_PROCESSOR=$TARGET_ARCH \
      -DTNN_ATLAS_ENABLE:BOOL="ON" \
      -DTNN_GLIBCXX_USE_CXX11_ABI_ENABLE=OFF \
      -DTNN_TORCHAIE_ENABLE:BOOL=$TORCHAIE

echo "Building TNN on ATLAS ..."
make -j $(nproc)


if [ -d ${TNN_INSTALL_DIR} ]
then
    rm -rf ${TNN_INSTALL_DIR}
fi
mkdir ${TNN_INSTALL_DIR}
mkdir ${TNN_INSTALL_DIR}/lib
mkdir ${TNN_INSTALL_DIR}/bin

cp -r ${TNN_ROOT_PATH}/include ${TNN_INSTALL_DIR}/
cp ${TNN_BUILD_DIR}/libTNN.so* ${TNN_INSTALL_DIR}/lib
cp ${TNN_BUILD_DIR}/test/TNNTest ${TNN_INSTALL_DIR}/bin

echo "Building TNN on ATLAS ... done!"
