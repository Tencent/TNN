#!/bin/bash

ABI="arm64-v8a"
CLEAN=""
WORK_DIR=`pwd`
FILTER=""
DEVICE_TYPE=""
KERNEL_TUNE="-et"
BUILD_DIR=build
ANDROID_DIR=/data/local/tmp/tnn-benchmark
OUTPUT_LOG_FILE=benchmark_layer_result.txt
LOOP_COUNT=10
ADB=adb

function usage() {
    echo "usage: ./benchmark_layer.sh  [-32] [-c] [-f] <filter-info> [-d] <device-id> [-t] <CPU/GPU>"
    echo "options:"
    echo "        -32         Build 32 bit."
    echo "        -c          Clean up build folders."
    echo "        -d          run with specified device"
    echo "        -f          specified layer"
    echo "        -t          CPU/GPU specify the platform to run"
    echo "        -et/-noet   set kernel enable tune on or off" 
}

function exit_with_msg() {
    echo $1
    exit 1
}

function clean_build() {
    echo $1 | grep "$BUILD_DIR\b" > /dev/null
    if [[ "$?" != "0" ]]; then
        exit_with_msg "Warnning: $1 seems not to be a BUILD folder."
    fi
    rm -rf $1
    mkdir $1
}

function build_android_bench() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR || exit
    cmake ../../.. \
          -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DANDROID_ABI="${ABI}" \
          -DANDROID_STL=c++_static \
          -DANDROID_NATIVE_API_LEVEL=android-14  \
          -DANDROID_TOOLCHAIN=clang \
          -DTNN_ARM_ENABLE:BOOL=ON \
          -DTNN_ARM82_ENABLE:BOOL=ON \
          -DTNN_OPENCL_ENABLE:BOOL=ON \
          -DTNN_TEST_ENABLE:BOOL=ON \
          -DTNN_BENCHMARK_MODE:BOOL=ON \
          -DTNN_UNIT_TEST_ENABLE:BOOL=ON \
          -DTNN_UNIT_TEST_BENCHMARK:BOOL=ON \
          -DTNN_PROFILER_ENABLE:BOOL=ON \
          -DBUILD_FOR_ANDROID_COMMAND=true
    make -j4
}

function bench_android() {
    build_android_bench

    if [ $? != 0 ];then
        exit_with_msg "build failed"
    fi

    $ADB shell "mkdir -p $ANDROID_DIR"
    find . -name "*.so" | while read solib; do
        $ADB push $solib  $ANDROID_DIR
    done
    $ADB push test/unit_test/unit_test $ANDROID_DIR/unit_test
    $ADB shell chmod 0777 $ANDROID_DIR/unit_test

    $ADB shell "getprop ro.product.model > ${ANDROID_DIR}/$OUTPUT_LOG_FILE"
    if [ "$DEVICE_TYPE" != "GPU" ] && [ "$DEVICE_TYPE" != "CPU" ];then
        DEVICE_TYPE=""
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "CPU" ];then
        $ADB shell "echo '\nbenchmark device: ARM \n' >> ${ANDROID_DIR}/$OUTPUT_LOG_FILE"
        $ADB shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=. ./unit_test ${KERNEL_TUNE} -ic ${LOOP_COUNT} -dt ARM --gtest_filter="*${FILTER}*" -ub >> $OUTPUT_LOG_FILE"
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "GPU" ];then
        LOOP_COUNT=1
        $ADB shell "echo '\nbenchmark device: OPENCL \n' >> ${ANDROID_DIR}/$OUTPUT_LOG_FILE"
        $ADB shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=. ./unit_test ${KERNEL_TUNE} -ic ${LOOP_COUNT} -dt OPENCL --gtest_filter="*${FILTER}*" -ub >> $OUTPUT_LOG_FILE"
    fi

    $ADB pull $ANDROID_DIR/$OUTPUT_LOG_FILE ../$OUTPUT_LOG_FILE
    cat ${WORK_DIR}/$OUTPUT_LOG_FILE
}

while [ "$1" != "" ]; do
    case $1 in
        -32)
            shift
            ABI="armeabi-v7a with NEON"
            ;;
        -c)
            shift
            CLEAN="-c"
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

bench_android
