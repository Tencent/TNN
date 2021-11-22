#!/bin/bash
ABI="arm64-v8a"
CLEAN=""
BUILD_ONLY=""
STL="c++_static"
WORK_DIR=`pwd`
BUILD_DIR=build_ut
FILTER=""
ANDROID_DIR=/data/local/tmp/unit_test
DUMP_DIR=$WORK_DIR/dump_data_unittest
ADB=adb

DEVICE_TYPE=""

function usage() {
    echo "usage: ./unit_test_android.sh  [-32] [-c] [-b] [-f] <filter> [-t] <CPU/GPU/HUAWEI_NPU>"
    echo "options:"
    echo "        -32   Build 32bit."
    echo "        -c    Clean up build folders."
    echo "        -b    build targets only"
    echo "        -f    filter"
    echo "        -t    CPU/GPU/HUAWEI_NPU specify the platform to run"
}
function die() {
    echo $1
    exit 1
}

function clean_build() {
    echo $1 | grep "$BUILD_DIR\b" > /dev/null
    if [[ "$?" != "0" ]]; then
        die "Warnning: $1 seems not to be a BUILD folder."
    fi
    rm -rf $1
    mkdir $1
}

function build_android() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    if [ "$DEVICE_TYPE" = "HUAWEI_NPU"  ]; then
        STL="c++_shared"
        HUAWEI_NPU="ON"

        #start to cp
        if [ ! -d ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/ ]; then
             mkdir -p ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/
        fi
        mkdir -p ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/armeabi-v7a
        mkdir -p ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/arm64-v8a
        cp $ANDROID_NDK/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a/libc++_shared.so  ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/armeabi-v7a/
        cp $ANDROID_NDK/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/arm64-v8a/
    fi
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR || exit
    cmake ../../../ \
          -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DANDROID_ABI="${ABI}" \
          -DANDROID_STL=${STL} \
          -DANDROID_NATIVE_API_LEVEL=android-14  \
          -DANDROID_TOOLCHAIN=clang \
          -DTNN_TEST_ENABLE:BOOL="ON"  \
          -DTNN_UNIT_TEST_ENABLE:BOOL="ON"  \
          -DTNN_ARM_ENABLE:BOOL="ON" \
          -DTNN_ARM82_ENABLE:BOOL="ON" \
          -DTNN_OPENCL_ENABLE:BOOL="ON" \
          -DTNN_HUAWEI_NPU_ENABLE:BOOL=$HUAWEI_NPU \
          -DBUILD_FOR_ANDROID_COMMAND=true
    make -j4
}

function run() {
    build_android
    if [ $? != 0 ]; then
        echo "build failed!"
        exit 0
    fi

    mkdir -p $DUMP_DIR

    $ADB shell "mkdir -p $ANDROID_DIR"
    find . -name "*.so" | while read solib; do
        $ADB push $solib  $ANDROID_DIR
    done

    $ADB push  test/unit_test/unit_test  $ANDROID_DIR
    $ADB shell chmod 0777 $ANDROID_DIR/unit_test

    $ADB shell "mkdir -p $ANDROID_DIR/dump_data"

    if [ "" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi

    $ADB shell "getprop ro.product.model > ${ANDROID_DIR}/test_log.txt"
    if [ "$DEVICE_TYPE" = "HUAWEI_NPU"  ]; then
        $ADB shell "echo 'Run Huawei Npu' >> $ANDROID_DIR/test_log.txt"
        $ADB shell "mkdir -p $ANDROID_DIR/lib"
        $ADB push $WORK_DIR/../../third_party/huawei_npu/cpp_lib/$ABI/* $ANDROID_DIR/lib
        $ADB push $WORK_DIR/../../third_party/huawei_npu/hiai_ddk_latest/$ABI/* $ANDROID_DIR/lib
        $ADB shell "cd $ANDROID_DIR; LD_LIBRARY_PATH=${ANDROID_DIR}/lib:$ANDROID_DIR ./unit_test -dt HUAWEI_NPU  --gtest_filter=\"*${FILTER}*\"  >> $ANDROID_DIR/test_log.txt"
    elif [ "$DEVICE_TYPE" = "CPU"  ]; then
        $ADB shell "echo 'Run ARM' >> $ANDROID_DIR/test_log.txt"
        $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./unit_test -dt ARM  --gtest_filter=\"*${FILTER}*\" >> $ANDROID_DIR/test_log.txt"
    elif [ "$DEVICE_TYPE" = "GPU"  ]; then
        $ADB shell "echo 'Run GPU' > $ANDROID_DIR/test_log.txt"
        $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./unit_test -dt OPENCL  --gtest_filter=\"*${FILTER}*\" >> $ANDROID_DIR/test_log.txt"
    else
        $ADB shell "echo 'Run ARM & GPU' >> $ANDROID_DIR/test_log.txt"
        $ADB shell "echo '===== ARM Unit Test =====' >> $ANDROID_DIR/test_log.txt"
        $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./unit_test -dt ARM  --gtest_filter=\"*${FILTER}*\" >> $ANDROID_DIR/test_log.txt"
        $ADB shell "echo '===== OPENCL Unit Test =====' >> $ANDROID_DIR/test_log.txt"
        $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./unit_test -dt OPENCL  --gtest_filter=\"*${FILTER}*\" >> $ANDROID_DIR/test_log.txt"
    fi
    $ADB pull $ANDROID_DIR/test_log.txt $DUMP_DIR
    cat $DUMP_DIR/test_log.txt
}

while [ "$1" != "" ]; do
    case $1 in
        -32)
            shift
            ABI="armeabi-v7a"
            ;;
        -c)
            shift
            CLEAN="-c"
            ;;
        -b)
            shift
            BUILD_ONLY="-b"
            ;;
        -f)
            shift
            FILTER=$1
            shift
            ;;
        -d)
            shift
            ADB="adb -s $1"
            shift
            ;;
        -t)
            shift
            DEVICE_TYPE="$1"
            shift
            ;;
        *)
            usage
            exit 1
    esac
done

run
