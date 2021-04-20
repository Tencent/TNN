#!/bin/bash

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

BUILD_DIR=${TNN_ROOT_PATH}/scripts/build_cuda_linux
TNN_INSTALL_DIR=${TNN_ROOT_PATH}/scripts/cuda_linux_release

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ${TNN_ROOT_PATH} \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_X86_ENABLE=OFF \
    -DTNN_CUDA_ENABLE=ON \
    -DTNN_TENSORRT_ENABLE=ON \
    -DTNN_BENCHMARK_MODE=OFF \
    -DTNN_BUILD_SHARED=ON \
    -DTNN_CONVERTER_ENABLE=OFF 

echo "Building TNN ..."
make -j32

if [ -d ${TNN_INSTALL_DIR} ]
then 
    rm -rf ${TNN_INSTALL_DIR}
fi
mkdir ${TNN_INSTALL_DIR}
mkdir ${TNN_INSTALL_DIR}/lib
mkdir ${TNN_INSTALL_DIR}/bin

cp -r ${TNN_ROOT_PATH}/include ${TNN_INSTALL_DIR}/
cp libTNN.so* ${TNN_INSTALL_DIR}/lib
cp test/TNNTest ${TNN_INSTALL_DIR}/bin

echo "Done"
