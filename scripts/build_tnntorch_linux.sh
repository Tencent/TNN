#!/bin/bash

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

TORCHVISION_ENABLE="OFF"
PYBIND_ENABLE="OFF"

export CUDNN_ROOT_DIR=/usr/local/cudnn-8.1.1
export TENSORRT_ROOT_DIR=/usr/local/TensorRT-7.2.3.4
if [ -z $1 ];then export LIBTORCH_ROOT_DIR=`find /usr/local/ -name "libtorch-shared-1.8.1+*"`
else
    export LIBTORCH_ROOT_DIR=$1
fi

export LIBTORCHVISION_ROOT_DIR=`find /usr/local/ -name "libtorchvision*-0.9.1+*"`

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
    -DTNN_TORCHVISION_ENABLE=${TORCHVISION_ENABLE} \
    -DTNN_PYBIND_ENABLE=${PYBIND_ENABLE} \
    -DTNN_GLIBCXX_USE_CXX11_ABI_ENABLE=OFF \
    -DTNN_TENSORRT_ENABLE=ON \
    -DTNN_BENCHMARK_MODE=OFF \
    -DTNN_BUILD_SHARED=ON \
    -DTNN_CONVERTER_ENABLE=OFF \
    -DTNN_TIACC_MODE=ON

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

# deps
cuda_dep_list=$( ldd libTNN.so | awk '{if (match($3, "/usr/local/cuda")){ print $3}}' )
cp $cuda_dep_list ${TNN_INSTALL_DIR}/lib/

# nvrtc
nvrtc_dep_list=$( find /usr/local/cuda/ -name "*nvrtc-builtins*" )
cp $nvrtc_dep_list ${TNN_INSTALL_DIR}/lib/

#cublas
cublas_dep_list=$( ldd libTNN.so | awk '{if (match($3, "cublas")){ print $3}}' )
cp $cublas_dep_list ${TNN_INSTALL_DIR}/lib/

#tensorrt
tensorrt_dep_list=$( ldd libTNN.so | awk '{if (match($3, "TensorRT")){ print $3}}' )
cp ${tensorrt_dep_list} ${TNN_INSTALL_DIR}/lib/
#tensorrt8 special 
tensorrt_builder_resource=`find ${TENSORRT_ROOT_DIR} -name "libnvinfer_builder_resource.so*"`
if [ -n "$tensorrt_builder_resource" ]; then  
    cp ${tensorrt_builder_resource} ${TNN_INSTALL_DIR}/lib/
fi

#cudnn
cudnn_dep_list=$( ldd libTNN.so | awk '{if (match($3, "cudnn")){ print $3}}' )
cp $cudnn_dep_list ${TNN_INSTALL_DIR}/lib/
cp ${CUDNN_ROOT_DIR}/lib64/libcudnn_cnn_infer.so.8 ${TNN_INSTALL_DIR}/lib/
cp ${CUDNN_ROOT_DIR}/lib64/libcudnn_ops_infer.so.8 ${TNN_INSTALL_DIR}/lib/

# torch
torch_dep_list=$( ldd libTNN.so | awk '{if (match($3,"libtorch-shared")){ print $3}}' )
cp $torch_dep_list ${TNN_INSTALL_DIR}/lib/

# torchvision libs
if [ "$TORCHVISION_ENABLE" = "ON" ]; then
    cp -d ${LIBTORCHVISION_ROOT_DIR}/lib/libtorchvision.so ${TNN_INSTALL_DIR}/lib/libtorchvision.so
fi

if [ "$PYBIND_ENABLE" = "ON" ]; then
    cp -d _pytnn.*.so ${TNN_INSTALL_DIR}/lib/
    cp ${TNN_ROOT_PATH}/source/pytnn/*.py ${TNN_INSTALL_DIR}/lib/
fi

echo Done
