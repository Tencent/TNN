#!/bin/bash

ABI="arm64-v8a"

BUILD_DIR=build
ANDROID_DIR=/data/local/tmp
LOOP_COUNT=10

function build_android_bench() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p build
    cd $BUILD_DIR
    cmake ../../.. \
          -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DANDROID_ABI="${ABI}" \
          -DANDROID_STL=c++_static \
          -DANDROID_NATIVE_API_LEVEL=android-14  \
          -DANDROID_TOOLCHAIN=clang \
          -DTNN_ARM_ENABLE:BOOL=ON \
          -DTNN_OPENCL_ENABLE:BOOL=ON \
          -DTNN_TEST_ENABLE:BOOL=ON \
          -DTNN_UNIT_TEST_ENABLE:BOOL=ON \
          -DTNN_BENCHMARK_OP:BOOL=ON \
          -DBUILD_FOR_ANDROID_COMMAND=true \
          -DNATIVE_LIBRARY_OUTPUT=.
    make -j4
}

function bench_android() {
    build_android_bench
    adb shell "mkdir -p $ANDROID_DIR"
    find . -name "*.so" | while read solib; do
        adb push $solib  $ANDROID_DIR
    done
    adb push test/unit_test/unit_test $ANDROID_DIR/unit_test
    adb shell chmod 0777 $ANDROID_DIR/unit_test

    adb shell "getprop ro.product.model > ${ANDROID_DIR}/benchmark_layer_result.txt"
    adb shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=. ./unit_test -ub 1 -ic ${LOOP_COUNT} -dt ARM $* >> benchmark_layer_result.txt"
    adb shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=. ./unit_test -ub 1 -ic ${LOOP_COUNT} -dt OPENCL $* >> benchmark_layer_result.txt"
    adb pull $ANDROID_DIR/benchmark_layer_result.txt ../benchmark_layer_result.txt
}

bench_android $*
