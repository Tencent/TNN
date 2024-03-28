#!/bin/bash

TNNTORCH="OFF"
ARM="ON"
OPENMP="ON"
OPENCL="OFF"
RKNPU="OFF"
CC=aarch64-linux-gnu-gcc
CXX=aarch64-linux-gnu-g++
TARGET_ARCH=aarch64

#export CUDNN_ROOT_DIR=/your/cudnn/dir/like/usr/local/cudnn-arm-8.9.3
#export TENSORRT_ROOT_DIR=/your/trt/dir/like/usr/local/TensorRT-arm-8.5.3.1

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

BUILD_DIR=${TNN_ROOT_PATH}/scripts/build_cuda_aarch64_linux
TNN_INSTALL_DIR=${TNN_ROOT_PATH}/scripts/cuda_aarch64_linux_release

TNN_VERSION_PATH=$TNN_ROOT_PATH/scripts/version
cd $TNN_VERSION_PATH
source $TNN_VERSION_PATH/version.sh
source $TNN_VERSION_PATH/add_version_attr.sh

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ${TNN_ROOT_PATH} \
    -DTNN_TEST_ENABLE=ON \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DTNN_ARM_ENABLE:BOOL=$ARM \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_CUDA_ENABLE=ON \
    -DTNN_TENSORRT_ENABLE=ON \
    -DTNN_TNNTORCH_ENABLE=${TNNTORCH} \
    -DTNN_RK_NPU_ENABLE:BOOL=$RKNPU \
    -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
    -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
    -DTNN_BENCHMARK_MODE=OFF \
    -DTNN_GLIBCXX_USE_CXX11_ABI_ENABLE=OFF \
    -DTNN_BUILD_SHARED=ON \
    -DTNN_CONVERTER_ENABLE=OFF \
    -DTNN_PYBIND_ENABLE=OFF

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
