#!/bin/bash

#export DDK_PATH=/data1/Ascend/ascend-toolkit/latest
#export NPU_HOST_LIB=/data1/Ascend/ascend-toolkit/latest/acllib/lib64

ARM="ON"
OPENMP="ON"
DEBUG=0
SHARED_LIB="ON"
BENCHMARK="OFF"
TNN_TEST="ON"
TARGET_ARCH=aarch64

TNN_BUILD_PATH=$PWD
if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi
TNN_VERSION_PATH=$TNN_ROOT_PATH/scripts/version
echo $TNN_ROOT_PATH
echo $TNN_VERSION_PATH
echo ' '
echo '******************** step 1: update version.h ********************'
cd $TNN_VERSION_PATH
source $TNN_VERSION_PATH/version.sh
source $TNN_VERSION_PATH/add_version_attr.sh

echo ' '
echo '******************** step 2: start build atlas ********************'
#删除旧SDK文件
cd $TNN_BUILD_PATH
if [ -x "build_atlas" ];then
rm -r build_atlas
fi

#新建build目录
mkdir build_atlas
cd build_atlas


cmake ${TNN_ROOT_PATH} \
      -DCMAKE_BUILD_TYPE=Release \
      -DDEBUG=$DEBUG \
      -DTNN_TEST_ENABLE:BOOL=$TNN_TEST \
      -DTNN_BENCHMARK_MODE:BOOL=$BENCHMARK \
      -DTNN_CPU_ENABLE:BOOL="ON"  \
      -DTNN_ARM_ENABLE:BOOL=$ARM \
      -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
      -DTNN_X86_ENABLE:BOOL="OFF"  \
      -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB \
      -DCMAKE_SYSTEM_PROCESSOR=$TARGET_ARCH \
      -DTNN_ATLAS_ENABLE:BOOL="ON"
make -j8

echo ' '
echo '******************** step 3: add version attr ********************'
#添加版本信息到库文件
cd $TNN_BUILD_PATH
if [ "$SHARED_LIB" = "ON" ];then
AddAllVersionAttr "$TNN_BUILD_PATH/build_atlas/libTNN.so"
AddAllVersionAttr "$TNN_BUILD_PATH/build64/libTNN.so"
else
AddAllVersionAttr "$TNN_BUILD_PATH/build_atlas/libTNN.a"
AddAllVersionAttr "$TNN_BUILD_PATH/build64/libTNN.a"
fi


echo '******************** step 4: copy to release ********************'
cd $TNN_BUILD_PATH
mkdir -p release_atlas
cd release_atlas
rm -rf *
mkdir lib
cd ..
if [ "$SHARED_LIB" = "ON" ];then
cp -d build_atlas/libTNN.so* release_atlas/lib
else
cp build_atlas/libTNN.a release_atlas/lib
fi
cp -r ${TNN_ROOT_PATH}/include release_atlas

echo "build done!"
