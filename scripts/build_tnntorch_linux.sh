#!/bin/bash

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

TORCHVISION_ENABLE="OFF"

export CUDNN_ROOT_DIR=/usr/local/cudnn-8.1.1
export TENSORRT_ROOT_DIR=/usr/local/TensorRT-7.2.3.4
export LIBTORCH_ROOT_DIR=/usr/local/libtorch-shared-1.8.1+cu102
export LIBTORCHVISION_ROOT_DIR=/usr/local/

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
    -DTNN_TORCHVISION_ENABLE=OFF \
    -DTNN_PYBIND_ENABLE=ON \
    -DTNN_GLIBCXX_USE_CXX11_ABI_ENABLE=OFF \
    -DTNN_TENSORRT_ENABLE=ON \
    -DTNN_BENCHMARK_MODE=OFF \
    -DTNN_BUILD_SHARED=ON \
    -DTNN_CONVERTER_ENABLE=OFF 

echo Building TNN ...
make -j8

if [ -d ${TNN_INSTALL_DIR} ]
then 
    rm -rf ${TNN_INSTALL_DIR}
fi

mkdir -p ${TNN_INSTALL_DIR}
mkdir -p ${TNN_INSTALL_DIR}/lib

echo $CUDA_TOOLKIT_ROOT_DIR

cp -r ${TNN_ROOT_PATH}/include ${TNN_INSTALL_DIR}/
cp -d libTNN.so* ${TNN_INSTALL_DIR}/lib/
cp -d _pytnn.*.so ${TNN_INSTALL_DIR}/lib/ 

# deps
cp -d /usr/local/cuda/lib64/libcudart.so* ${TNN_INSTALL_DIR}/lib/
cp -d /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so* ${TNN_INSTALL_DIR}/lib/
cp -d /usr/local/cuda/targets/x86_64-linux/lib/libnvToolsExt*.so* ${TNN_INSTALL_DIR}/lib/
cp -d /usr/lib/x86_64-linux-gnu/libcuda.so.* ${TNN_INSTALL_DIR}/lib/
cp -d /usr/lib/x86_64-linux-gnu/libcublas*.so* ${TNN_INSTALL_DIR}/lib/
cp -d $TENSORRT_ROOT_DIR/lib/libnvinfer.so* ${TNN_INSTALL_DIR}/lib/
cp -d $TENSORRT_ROOT_DIR/lib/libnvinfer_plugin.so* ${TNN_INSTALL_DIR}/lib/
cp -d $TENSORRT_ROOT_DIR/lib/libmyelin.so* ${TNN_INSTALL_DIR}/lib/
cp -d $CUDNN_ROOT_DIR/lib64/libcudnn.so* ${TNN_INSTALL_DIR}/lib/
cp -d $CUDNN_ROOT_DIR/lib64/libcudnn_cnn_infer.so* ${TNN_INSTALL_DIR}/lib/
cp -d $CUDNN_ROOT_DIR/lib64/libcudnn_ops_infer.so* ${TNN_INSTALL_DIR}/lib/

# torch
cp -d ${LIBTORCH_ROOT_DIR}/lib/libtorch.so ${TNN_INSTALL_DIR}/lib/
cp -d ${LIBTORCH_ROOT_DIR}/lib/libtorch_cpu.so ${TNN_INSTALL_DIR}/lib/
cp -d ${LIBTORCH_ROOT_DIR}/lib/libtorch_cuda.so ${TNN_INSTALL_DIR}/lib/
cp -d ${LIBTORCH_ROOT_DIR}/lib/libc10.so ${TNN_INSTALL_DIR}/lib/
cp -d ${LIBTORCH_ROOT_DIR}/lib/libc10_cuda.so ${TNN_INSTALL_DIR}/lib/
cp -d ${LIBTORCH_ROOT_DIR}/lib/libgomp*.so* ${TNN_INSTALL_DIR}/lib/
cp -d ${LIBTORCH_ROOT_DIR}/lib/libnvToolsExt*.so* ${TNN_INSTALL_DIR}/lib/
cp -d ${LIBTORCH_ROOT_DIR}/lib/libcudart-*.so* ${TNN_INSTALL_DIR}/lib/

# torchvision libs
if [ "$TORCHVISION_ENABLE" = "ON" ]; then
    cp -d ${LIBTORCHVISION_ROOT_DIR}/lib/libtorchvision.so ${TNN_INSTALL_DIR}/lib/libtorchvision.so
fi

cp ${TNN_ROOT_PATH}/source/pytnn/*.py ${TNN_INSTALL_DIR}/lib/

echo Done
