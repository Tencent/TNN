#!/bin/bash

ABIA32="armeabi-v7a with NEON"
#ABIA32="armeabi-v7a"
ABIA64="arm64-v8a"
STL="c++_static"
#STL="gnustl_static"
SHARED_LIB="ON"
ARM="ON"
OPENMP="ON"
OPENCL="ON"
BENMARK_MODE="OFF"
DEBUG="OFF"
SHARING_MEM_WITH_OPENGL=0

# check ANDROID_NDK whether set.
if [ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]; then
   echo -e "Not found: build/cmake/android.toolchain.cmake in ANDROID_NDK:$ANDROID_NDK"
   echo -e "Please download android ndk and set ANDROID_NDK environment variable."
   exit -1
fi

TNN_BUILD_PATH=$PWD
if [ -z $TNN_ROOT_PATH ]
then
      TNN_ROOT_PATH=`git rev-parse --show-toplevel`
fi
TNN_VERSION_PATH=$TNN_ROOT_PATH/scripts/version
echo $TNN_ROOT_PATH
echo $TNN_VERSION_PATH
echo $ABI
echo ' '
echo '******************** step 1: update version.h ********************'
cd $TNN_VERSION_PATH
source $TNN_VERSION_PATH/version.sh
source $TNN_VERSION_PATH/add_version_attr.sh

echo ' '
echo '******************** step 2: start build rpn arm32 ********************'
#删除旧SDK文件
cd $TNN_BUILD_PATH
if [ -x "build32" ];then
rm -r build32
fi

#新建build目录
mkdir build32
cd build32
echo $ABIA32
cmake ${TNN_ROOT_PATH} \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DDEBUG:BOOL=$DEBUG \
      -DANDROID_ABI="${ABIA32}" \
      -DANDROID_STL=${STL} \
      -DANDROID_NATIVE_API_LEVEL=android-14  \
      -DANDROID_TOOLCHAIN=clang \
      -DBUILD_FOR_ANDROID_COMMAND=true \
      -DTNN_ARM_ENABLE:BOOL=$ARM \
      -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
      -DTNN_BENCHMARK_MODE:BOOL=$BENMARK_MODE \
      -DTNN_TEST_ENABLE:BOOL=ON \
      -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
      -DSHARING_MEM_WITH_OPENGL=${SHARING_MEM_WITH_OPENGL} \
      -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB
make -j32

echo ' '
echo '******************** step 3: start build rpn arm64 ********************'
#删除旧SDK文件
cd $TNN_BUILD_PATH
if [ -x "build64" ];then
rm -r build64
fi

#新建build目录
mkdir build64
cd build64
echo $ABIA64
cmake ${TNN_ROOT_PATH} \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DDEBUG:BOOL=$DEBUG \
      -DANDROID_ABI="${ABIA64}" \
      -DANDROID_STL=${STL} \
      -DANDROID_NATIVE_API_LEVEL=android-14  \
      -DANDROID_TOOLCHAIN=clang \
      -DBUILD_FOR_ANDROID_COMMAND=true \
      -DTNN_ARM_ENABLE:BOOL=$ARM \
      -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
      -DTNN_TEST_ENABLE:BOOL=ON \
      -DTNN_BENCHMARK_MODE:BOOL=$BENMARK_MODE \
      -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
      -DSHARING_MEM_WITH_OPENGL=${SHARING_MEM_WITH_OPENGL} \
      -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB
make -j32

echo ' '
echo '******************** step 4: add version attr ********************'
#添加版本信息到库文件
cd $TNN_BUILD_PATH
if [ "$SHARED_LIB" = "ON" ];then
AddAllVersionAttr "$TNN_BUILD_PATH/build32/libTNN.so"
AddAllVersionAttr "$TNN_BUILD_PATH/build64/libTNN.so"
else
AddAllVersionAttr "$TNN_BUILD_PATH/build32/libTNN.a"
AddAllVersionAttr "$TNN_BUILD_PATH/build64/libTNN.a"
fi


echo '******************** step 4: copy to release ********************'
cd $TNN_BUILD_PATH
mkdir -p release
cd release
rm -rf *
mkdir armeabi-v7a
mkdir arm64-v8a
cd ..
if [ "$SHARED_LIB" = "ON" ];then
cp build32/libTNN.so release/armeabi-v7a
cp build64/libTNN.so release/arm64-v8a
else
cp build32/libTNN.a release/armeabi-v7a
cp build64/libTNN.a release/arm64-v8a
fi
cp -r ${TNN_ROOT_PATH}/include release

echo "build done!"
