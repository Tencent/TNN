#!/bin/bash

CLEAN=""
WORK_DIR=`pwd`
FILTER=""
DEVICE_TYPE=""
KERNEL_TUNE="-et"
BUILD_DIR=build
OUTPUT_LOG_FILE=benchmark_layer_result.txt
LOOP_COUNT=10

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

function build_linux_bench() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p build
    cd $BUILD_DIR
    cmake ../../.. \
          -DCMAKE_BUILD_TYPE=Release \
          -DTNN_ARM_ENABLE:BOOL=ON \
          -DTNN_OPENCL_ENABLE:BOOL=ON \
          -DTNN_TEST_ENABLE:BOOL=ON \
          -DTNN_BENCHMARK_MODE:BOOL=ON \
          -DTNN_UNIT_TEST_ENABLE:BOOL=ON \
          -DTNN_UNIT_TEST_BENCHMARK:BOOL=ON \
          -DTNN_PROFILER_ENABLE:BOOL=ON
    make -j4
}

function bench_android() {
    build_linux_bench

    if [ $? != 0 ];then
        exit_with_msg "build failed"
    fi

    if [ "$DEVICE_TYPE" != "GPU" ] && [ "$DEVICE_TYPE" != "CPU" ];then
        DEVICE_TYPE=""
    fi

    echo 'layer benchmark' 2>&1 |tee $WORK_DIR/$OUTPUT_LOG_FILE
    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "CPU" ];then
        echo 'benchmark device: ARM' 2>&1 |tee -a $WORK_DIR/$OUTPUT_LOG_FILE
        cd ${WORK_DIR}; LD_LIBRARY_PATH=. ./build/test/unit_test/unit_test ${KERNEL_TUNE} -ic $LOOP_COUNT -dt ARM --gtest_filter="*${FILTER}*" -ub 2>&1 |tee -a $WORK_DIR/$OUTPUT_LOG_FILE
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "GPU" ];then
        LOOP_COUNT=1
        echo 'benchmark device: OPENCL' 2>&1 |tee -a $WORK_DIR/$OUTPUT_LOG_FILE
        cd ${WORK_DIR}; LD_LIBRARY_PATH=. ./build/test/unit_test/unit_test ${KERNEL_TUNE} -ic $LOOP_COUNT -dt OPENCL --gtest_filter="*${FILTER}*" -ub 2>&1 |tee -a $WORK_DIR/$OUTPUT_LOG_FILE
    fi
}

while [ "$1" != "" ]; do
    case $1 in
        -c)
            shift
            CLEAN="-c"
            ;;
        -f)
            shift
            FILTER=$1
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
