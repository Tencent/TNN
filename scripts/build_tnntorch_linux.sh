#!/bin/bash

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

export CUDNN_ROOT_DIR=/usr/local/cudnn-8.1.1
export TENSORRT_ROOT_DIR=/usr/local/TensorRT-7.2.3.4
export LIBTORCH_ROOT_DIR=/usr/local/libtorch-cxx11-abi-1.8.1+cu102
export LIBTORCHVISION_ROOT_DIR=/usr/local/libtorchvision-cxx11-abi-0.9.1+cu102

BUILD_DIR=${TNN_ROOT_PATH}/scripts/build_tnntorch_linux
TNN_INSTALL_DIR=${TNN_ROOT_PATH}/scripts/tnntorch_linux_release

TNN_VERSION_PATH=$TNN_ROOT_PATH/scripts/version
cd $TNN_VERSION_PATH
source $TNN_VERSION_PATH/version.sh
source $TNN_VERSION_PATH/add_version_attr.sh

# rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ${TNN_ROOT_PATH} \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_X86_ENABLE=ON \
    -DTNN_CUDA_ENABLE=ON \
    -DTNN_TNNTORCH_ENABLE=ON \
    -DTNN_TENSORRT_ENABLE=ON \
    -DTNN_BENCHMARK_MODE=OFF \
    -DTNN_BUILD_SHARED=ON \
    -DTNN_CONVERTER_ENABLE=OFF 

echo "Building TNN ..."
# make -j

# if [ -d ${TNN_INSTALL_DIR} ]
# then 
#     rm -rf ${TNN_INSTALL_DIR}
# fi

# mkdir ${TNN_INSTALL_DIR}
# mkdir ${TNN_INSTALL_DIR}/lib
# mkdir ${TNN_INSTALL_DIR}/bin

# cp -r ${TNN_ROOT_PATH}/include ${TNN_INSTALL_DIR}/
# cp libTNN.so* ${TNN_INSTALL_DIR}/lib
# cp test/TNNTest ${TNN_INSTALL_DIR}/bin

echo "Done"
