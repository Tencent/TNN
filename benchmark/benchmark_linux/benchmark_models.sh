#!/bin/bash

PROFILING="OFF"
CLEAN=""
DEVICE_TYPE=""
MODEL_TYPE=TNN
USE_NCNN_MODEL=0
SHARED_LIB="ON"
OPENCL="ON"

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/../..
fi

WORK_DIR=`pwd`
BENCHMARK_MODEL_DIR=$WORK_DIR/../benchmark-model
BUILD_DIR=build
OUTPUT_LOG_FILE=benchmark_models_result.txt
LOOP_COUNT=20
WARM_UP_COUNT=10

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

function build_bench() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p build
    cd $BUILD_DIR
    cmake ${TNN_ROOT_PATH} \
        -DCMAKE_BUILD_TYPE=Release \
        -DTNN_CPU_ENABLE=ON \
        -DTNN_X86_ENABLE=ON \
        -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
        -DTNN_PROFILER_ENABLE:BOOL=${PROFILING} \
        -DTNN_TEST_ENABLE=ON \
        -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB \
        -DTNN_BENCHMARK_MODE=ON \
        -DINTTYPES_FORMAT=C99

    make -j4
}

function bench_linux() {
    build_bench
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

    echo "benchmark log:" > $WORK_DIR/log_temp.txt
    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "CPU" ];then
        device=X86
        echo "benchmark device: ${device} " >> $WORK_DIR/log_temp.txt

        for benchmark_model in ${benchmark_model_list[*]}
        do
            cd ${WORK_DIR}; LD_LIBRARY_PATH=. ./build/test/TNNTest -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mt ${MODEL_TYPE} -mp ${BENCHMARK_MODEL_DIR}/${benchmark_model}  >> log_temp.txt
        done
    fi

    if [ "ON" == $PROFILING ]; then
        WARM_UP_COUNT=5
        LOOP_COUNT=1
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "GPU" ];then
        device=OPENCL
        echo "benchmark device: ${device} " >> $WORK_DIR/log_temp.txt
        for benchmark_model in ${benchmark_model_list[*]}
        do
            cd ${WORK_DIR}; LD_LIBRARY_PATH=. ./build/test/TNNTest -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mt ${MODEL_TYPE} -mp ${BENCHMARK_MODEL_DIR}/${benchmark_model} -et >> log_temp.txt
        done
    fi

    cat $WORK_DIR/log_temp.txt |grep "time cost:" > $WORK_DIR/$OUTPUT_LOG_FILE
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

bench_linux
