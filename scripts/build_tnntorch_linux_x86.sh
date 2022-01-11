#!/bin/bash

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

TORCHVISION_ENABLE="ON"
PYBIND_ENABLE="OFF"

export CUDNN_ROOT_DIR=/usr/local/cudnn-8.1.1
export TENSORRT_ROOT_DIR=/usr/local/TensorRT-7.2.3.4
if [ -z $1 ];then export LIBTORCH_ROOT_DIR=`find /usr/local/ -name "libtorch-shared-1.8.1+*"`
else
    export LIBTORCH_ROOT_DIR=$1
fi

export LIBTORCHVISION_ROOT_DIR=`find /usr/local/ -name "libtorchvision*-0.9.1+*"`

BUILD_DIR=${TNN_ROOT_PATH}/scripts/build_tnntorch_linux_x86
TNN_INSTALL_DIR=${TNN_ROOT_PATH}/scripts/tnntorch_linux_x86_release

TNN_VERSION_PATH=$TNN_ROOT_PATH/scripts/version
cd $TNN_VERSION_PATH
source $TNN_VERSION_PATH/version.sh
source $TNN_VERSION_PATH/add_version_attr.sh

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# add openvino
OPENVINO_BUILD_SHARED="ON"

OPENVINO_INSTALL_PATH=${BUILD_DIR}/../openvinoInstallShared
if [ "${OPENVINO_BUILD_SHARED}" = "OFF" ]
then
    OPENVINO_INSTALL_PATH=${BUILD_DIR}/../openvinoInstallStatic
fi

export OPENVINO_ROOT_DIR=${OPENVINO_INSTALL_PATH}
export GIT_LFS_SKIP_SMUDGE=1

# check_cmake() {
#     if !(command -v cmake > /dev/null 2>&1); then
#         echo "Cmake not found!"
#         exit 1
#     fi

#     for var in $(cmake --version | awk 'NR==1{print $3}')
#     do
#         cmake_version=$var
#     done
#     function version_lt { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"; }

#     if (version_lt $cmake_version 3.11); then
#         echo "Cmake 3.11 or higher is required. You are running version ${cmake_version}"
#         exit 2
#     fi
# }

# clone_openvino() {
#     mkdir -p ${BUILD_DIR}
#     cd ${BUILD_DIR}

#     if [ ! -d openvino ]
#     then
#         git clone --recursive https://github.com/openvinotoolkit/openvino.git
#     fi
#     cd openvino
#     git reset --hard 18e83a2
#     git submodule update --init --recursive
#     #sed -i '152 i /*' inference-engine/src/mkldnn_plugin/nodes/reduce.cpp
#     #sed -i '157 i */' inference-engine/src/mkldnn_plugin/nodes/reduce.cpp

#     # 编译静态库
#     if [ "${OPENVINO_BUILD_SHARED}" = "OFF" ]
#     then
#         sed -i '152,152s/SHARED/STATIC/g' inference-engine/src/inference_engine/CMakeLists.txt
#         sed -i 's/SHARED/STATIC/g' inference-engine/src/legacy_api/CMakeLists.txt
#         sed -i 's/SHARED/STATIC/g' inference-engine/src/transformations/CMakeLists.txt
#         sed -i 's/SHARED/STATIC/g' inference-engine/src/low_precision_transformations/CMakeLists.txt
#         sed -i 's/SHARED/STATIC/g' ngraph/src/ngraph/CMakeLists.txt
#     fi
# }

# build_openvino() {

#     if [ ! -d ${OPENVINO_INSTALL_PATH} ]
#     then
#         cd ${BUILD_DIR}/openvino
#         mkdir -p build && cd build
#         echo "Configuring Openvino ..."
#         cmake ../ \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DENABLE_OPENCV=OFF \
#         -DCMAKE_INSTALL_PREFIX=${OPENVINO_INSTALL_PATH} \
#         -DENABLE_TBB_RELEASE_ONLY=OFF \
#         -DTHREADING=TBB_AUTO \
#         -DNGRAPH_COMPONENT_PREFIX="deployment_tools/ngraph/" \
#         -DENABLE_MYRIAD=OFF \
#         -DENABLE_CLDNN=OFF \
#         -DENABLE_GNA=OFF \
#         -DENABLE_VPU=OFF \
#         -DENABLE_SAMPLES=OFF \
#         -DNGRAPH_JSON_ENABLE=OFF \
#         -DENABLE_SPEECH_DEMO=OFF \
#         -DNGRAPH_ONNX_IMPORT_ENABLE=OFF \
#         -DENABLE_PROFILING_ITT=OFF \
#         -DTREAT_WARNING_AS_ERROR=OFF \

#         echo "Building Openvino ..."
#         make -j7
#         make install
#     fi
# }

copy_openvino_libraries() {

    local LIB_EXT=".so"
    if [ "${OPENVINO_BUILD_SHARED}" = "OFF" ]
    then
        LIB_EXT=".a"
    fi

    cd ${BUILD_DIR}

    if [ -d ${OPENVINO_INSTALL_PATH}/deployment_tools/ngraph/lib64/ ]
    then
        mkdir -p ${OPENVINO_INSTALL_PATH}/deployment_tools/ngraph/lib
        cp ${OPENVINO_INSTALL_PATH}/deployment_tools/ngraph/lib64/libngraph${LIB_EXT} ${OPENVINO_INSTALL_PATH}/deployment_tools/ngraph/lib/
    fi

    # if [ -d ${OPENVINO_INSTALL_PATH}/lib64/ ]
    # then
    #     mkdir -p ${OPENVINO_INSTALL_PATH}/lib
    #     cp ${OPENVINO_INSTALL_PATH}/lib64/libpugixml.a ${OPENVINO_INSTALL_PATH}/lib/
    # fi

    if [ ! -d ${TNN_INSTALL_DIR} ] 
    then
        mkdir -p ${TNN_INSTALL_DIR}
    fi

    if [ ! -d ${TNN_INSTALL_DIR}/bin ] 
    then
        mkdir -p ${TNN_INSTALL_DIR}/bin
    fi

    if [ ! -d ${TNN_INSTALL_DIR}/lib ] 
    then
        mkdir -p ${TNN_INSTALL_DIR}/lib
    fi

    cp ${OPENVINO_INSTALL_PATH}/deployment_tools/inference_engine/lib/intel64/plugins.xml ${TNN_INSTALL_DIR}/lib
    cp ${OPENVINO_INSTALL_PATH}/deployment_tools/inference_engine/lib/intel64/plugins.xml ${BUILD_DIR}/
    cp ${OPENVINO_INSTALL_PATH}/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so ${TNN_INSTALL_DIR}/lib/
    cp ${OPENVINO_INSTALL_PATH}/deployment_tools/inference_engine/external/tbb/lib/* ${TNN_INSTALL_DIR}/lib/


    if [ "${OPENVINO_BUILD_SHARED}" = "ON" ]
    then
        cp ${OPENVINO_INSTALL_PATH}/deployment_tools/inference_engine/lib/intel64/libinference_engine${LIB_EXT} ${TNN_INSTALL_DIR}/lib/
        cp ${OPENVINO_INSTALL_PATH}/deployment_tools/inference_engine/lib/intel64/libinference_engine_legacy${LIB_EXT} ${TNN_INSTALL_DIR}/lib/
        cp ${OPENVINO_INSTALL_PATH}/deployment_tools/inference_engine/lib/intel64/libinference_engine_transformations${LIB_EXT} ${TNN_INSTALL_DIR}/lib/
        cp ${OPENVINO_INSTALL_PATH}/deployment_tools/inference_engine/lib/intel64/libinference_engine_lp_transformations${LIB_EXT} ${TNN_INSTALL_DIR}/lib/
        cp ${OPENVINO_INSTALL_PATH}/deployment_tools/ngraph/lib/libngraph${LIB_EXT} ${TNN_INSTALL_DIR}/lib/
    fi
}

# clone_openvino

# build_openvino

# add openvino end

# rm -rf ${BUILD_DIR}

cmake ${TNN_ROOT_PATH} \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_X86_ENABLE=ON \
    -DTNN_CUDA_ENABLE=ON \
    -DTNN_TNNTORCH_ENABLE=ON \
    -DTNN_OPENVINO_ENABLE=ON \
    -DTNN_OPENVINO_BUILD_SHARED=${OPENVINO_BUILD_SHARED} \
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

copy_openvino_libraries

# deps
cuda_dep_list=$( ldd libTNN.so | awk '{if (match($3, "/usr/local/cuda")){ print $3}}' )
cp $cuda_dep_list ${TNN_INSTALL_DIR}/lib/

#cublas
cublas_dep_list=$( ldd libTNN.so | awk '{if (match($3, "cublas")){ print $3}}' )
cp $cublas_dep_list ${TNN_INSTALL_DIR}/lib/

#tensorrt
tensorrt_dep_list=$( ldd libTNN.so | awk '{if (match($3, "TensorRT")){ print $3}}' )
cp ${tensorrt_dep_list} ${TNN_INSTALL_DIR}/lib/

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
