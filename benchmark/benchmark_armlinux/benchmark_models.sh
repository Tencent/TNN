#!/bin/bash

PROFILING="OFF"
CLEAN=""
DEVICE_TYPE=""
MODEL_TYPE=TNN
USE_NCNN_MODEL=0
SHARED_LIB="ON"
ARM="ON"
OPENMP="ON"
OPENCL="ON"
CC=aarch64-linux-gnu-gcc
CXX=aarch64-linux-gnu-g++
TARGET_ARCH=aarch64

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/../..
fi

WORK_DIR=`pwd`
BENCHMARK_MODEL_DIR=$WORK_DIR/../benchmark-model
BUILD_DIR=build
OUTPUT_LOG_FILE=benchmark_models_result.txt
LOOP_COUNT=16
WARM_UP_COUNT=8

benchmark_model_list=(
#test.tnnproto \
)

function usage() {
    echo "usage: ./benchmark_models.sh  [-32] [-c] [-b] [-f] [-t] <CPU/GPU>"
    echo "options:"
    echo "        -32   Build 32 bit."
    echo "        -c    Clean up build folders."
    echo "        -b    build targets only"
    echo "        -f    build profiling targets "
    echo "        -t    CPU/GPU specify the platform to run"
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

function build_armlinux_bench() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p build
    cd $BUILD_DIR
    cmake ${TNN_ROOT_PATH} \
        -DCMAKE_SYSTEM_NAME=Linux  \
        -DTNN_TEST_ENABLE=ON \
        -DTNN_CPU_ENABLE=ON \
        -DCMAKE_C_COMPILER=$CC \
        -DCMAKE_CXX_COMPILER=$CXX \
        -DCMAKE_BUILD_TYPE=Debug \
        -DTNN_CPU_ENABLE:BOOL=ON \
        -DTNN_ARM_ENABLE:BOOL=$ARM \
        -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
        -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
        -DTNN_PROFILER_ENABLE:BOOL=${PROFILING} \
        -DTNN_TEST_ENABLE=ON \
        -DTNN_UNIT_TEST_ENABLE=ON \
        -DTNN_COVERAGE=ON \
        -DCMAKE_SYSTEM_PROCESSOR=$TARGET_ARCH \
        -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB \
        -DTNN_BENCHMARK_MODE=ON 

    make -j4
}

function bench_armlinux() {
    build_armlinux_bench
    if [ $? != 0 ];then
        exit_with_msg "build failed"
    fi

    if [ "" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi

    cd ${BENCHMARK_MODEL_DIR}

    if [ ${#benchmark_model_list[*]} == 0 ];then
        benchmark_model_list=`ls *.tnnproto`
    fi

    if [ "$DEVICE_TYPE" != "GPU" ] && [ "$DEVICE_TYPE" != "CPU" ];then
        DEVICE_TYPE=""
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "CPU" ];then
        device=ARM
        echo "benchmark device: ${device} " >> $WORK_DIR/$OUTPUT_LOG_FILE

        for benchmark_model in ${benchmark_model_list[*]}
        do
            cd ${WORK_DIR}; LD_LIBRARY_PATH=. ./build/test/TNNTest -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mt ${MODEL_TYPE} -mp ${BENCHMARK_MODEL_DIR}/${benchmark_model}  >> $OUTPUT_LOG_FILE
        done
    fi

    if [ "ON" == $PROFILING ]; then
        WARM_UP_COUNT=5
        LOOP_COUNT=1
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "GPU" ];then
        device=OPENCL
        echo "benchmark device: ${device} " >> $WORK_DIR/$OUTPUT_LOG_FILE
        for benchmark_model in ${benchmark_model_list[*]}
        do
            cd ${WORK_DIR}; LD_LIBRARY_PATH=. ./build/test/TNNTest -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mt ${MODEL_TYPE} -mp ${BENCHMARK_MODEL_DIR}/${benchmark_model}  >> $OUTPUT_LOG_FILE
        done
    fi

    echo '' >> $OUTPUT_LOG_FILE
    date  >> $OUTPUT_LOG_FILE

    cat ${WORK_DIR}/$OUTPUT_LOG_FILE
}

while [ "$1" != "" ]; do
    case $1 in
        -32)
            shift
            CC=arm-linux-gnueabihf-gcc
            CXX=arm-linux-gnueabihf-g++
            TARGET_ARCH=arm
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
            PROFILING="ON"
            ;;
        -t)
            shift
            DEVICE_TYPE="$1"
            shift
            ;;
        -n)
            shift
            MODEL_TYPE=NCNN
            ;;
        *)
            usage
            exit 1
    esac
done

bench_armlinux
