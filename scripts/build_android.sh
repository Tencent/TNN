#!/bin/bash

#ABIA32="armeabi-v7a with NEON"
ABIA32="armeabi-v7a"
ABIA64="arm64-v8a"
STL="c++_static"
#STL="gnustl_static"
SHARED_LIB="ON"
ARM="ON"
ARM82="ON"
OPENMP="ON"
OPENCL="ON"
#HUAWEI_NPU="ON"
if [ -z "$HUAWEI_NPU" ]; then
    HUAWEI_NPU="OFF"
fi
BENMARK_MODE="OFF"
DEBUG="OFF"
INCREMENTAL_COMPILE="OFF"
SHARING_MEM_WITH_OPENGL=0
ANDROID_API_LEVEL="android-14"
# check ANDROID_NDK whether set.
if [ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]; then
   echo -e "Not found: build/cmake/android.toolchain.cmake in ANDROID_NDK:$ANDROID_NDK"
   echo -e "Please download android ndk and set ANDROID_NDK environment variable."
   exit -1
fi

while [ "$1" != "" ]; do
    case $1 in
        -ic)
            shift
            INCREMENTAL_COMPILE="ON"
            ;;
        *)
            usage
            exit 1
    esac
done


TNN_BUILD_PATH=$PWD
if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
    echo $TNN_ROOT_PATH
fi

if [ "$HUAWEI_NPU" == "ON" ]
then
    echo "NPU Enable"
    # set c++ shared
    STL="c++_shared"
    #start to cp
    if [ ! -d ${TNN_ROOT_PATH}/third_party/huawei_npu/cpp_lib/ ]; then
         mkdir -p ${TNN_ROOT_PATH}/third_party/huawei_npu/cpp_lib/
    fi
    mkdir -p ${TNN_ROOT_PATH}/third_party/huawei_npu/cpp_lib/armeabi-v7a
    mkdir -p ${TNN_ROOT_PATH}/third_party/huawei_npu/cpp_lib/arm64-v8a
    cp $ANDROID_NDK/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a/libc++_shared.so ${TNN_ROOT_PATH}/third_party/huawei_npu/cpp_lib/armeabi-v7a/
    cp $ANDROID_NDK/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so ${TNN_ROOT_PATH}/third_party/huawei_npu/cpp_lib/arm64-v8a/
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
cd $TNN_BUILD_PATH
if [ -x "build32" ];then
    if [ "${INCREMENTAL_COMPILE}" = "OFF" ];then
        echo 'remove build32'
        rm -r build32
        mkdir build32
    fi
else
    mkdir -p build32
fi

cd build32
echo $ABIA32
cmake ${TNN_ROOT_PATH} \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DDEBUG:BOOL=$DEBUG \
      -DANDROID_ABI="${ABIA32}" \
      -DANDROID_STL=${STL} \
      -DANDROID_NATIVE_API_LEVEL=${ANDROID_API_LEVEL}  \
      -DANDROID_TOOLCHAIN=clang \
      -DBUILD_FOR_ANDROID_COMMAND=true \
      -DTNN_CPU_ENABLE:BOOL=ON \
      -DTNN_ARM_ENABLE:BOOL=$ARM \
      -DTNN_ARM82_ENABLE:BOOL=$ARM82 \
      -DTNN_HUAWEI_NPU_ENABLE:BOOL=$HUAWEI_NPU \
      -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
      -DTNN_BENCHMARK_MODE:BOOL=$BENMARK_MODE \
      -DTNN_TEST_ENABLE:BOOL=ON \
      -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
      -DSHARING_MEM_WITH_OPENGL=${SHARING_MEM_WITH_OPENGL} \
      -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB
make -j8

# check ret code for ci
if [ 0 -ne $? ]
then
  exit -1
fi

echo ' '
echo '******************** step 3: start build rpn arm64 ********************'
cd $TNN_BUILD_PATH
if [ -x "build64" ];then
    if [ "${INCREMENTAL_COMPILE}" = "OFF" ];then
        echo 'remove build64'
        rm -r build64
        mkdir build64
    fi
else
    mkdir -p build64
fi

cd build64
echo $ABIA64
cmake ${TNN_ROOT_PATH} \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DDEBUG:BOOL=$DEBUG \
      -DANDROID_ABI="${ABIA64}" \
      -DANDROID_STL=${STL} \
      -DANDROID_NATIVE_API_LEVEL=${ANDROID_API_LEVEL}  \
      -DANDROID_TOOLCHAIN=clang \
      -DBUILD_FOR_ANDROID_COMMAND=true \
      -DTNN_CPU_ENABLE:BOOL=ON \
      -DTNN_ARM_ENABLE:BOOL=$ARM \
      -DTNN_ARM82_ENABLE:BOOL=$ARM82 \
      -DTNN_HUAWEI_NPU_ENABLE:BOOL=$HUAWEI_NPU \
      -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
      -DTNN_TEST_ENABLE:BOOL=ON \
      -DTNN_BENCHMARK_MODE:BOOL=$BENMARK_MODE \
      -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
      -DSHARING_MEM_WITH_OPENGL=${SHARING_MEM_WITH_OPENGL} \
      -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB
make -j8

# check ret code for ci
if [ 0 -ne $? ]
then
  exit -1
fi

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
mkdir -p armeabi-v7a
mkdir -p arm64-v8a
cd ..
if [ "$SHARED_LIB" = "ON" ];then
    cp build32/libTNN.so release/armeabi-v7a
    cp build64/libTNN.so release/arm64-v8a
else
    cp build32/libTNN.a release/armeabi-v7a
    cp build64/libTNN.a release/arm64-v8a
fi
cp -r ${TNN_ROOT_PATH}/include release
if [  "$HUAWEI_NPU" == "ON" ]; then
    cp ${TNN_ROOT_PATH}/third_party/huawei_npu/hiai_ddk_latest/armeabi-v7a/* release/armeabi-v7a/
    cp ${TNN_ROOT_PATH}/third_party/huawei_npu/hiai_ddk_latest/arm64-v8a/* release/arm64-v8a/
fi
echo "build done!"

if [ "$SHARED_LIB" != "ON" ]; then
    echo -e "\033[31m[WARNING] TNN is built as a static library, link it with option -Wl,--whole-archive tnn -Wl,--no-whole-archive\033[0m"
fi
