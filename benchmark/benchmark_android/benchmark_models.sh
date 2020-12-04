#!/bin/bash

export PATH=$PATH:$ANDROID_HOME/platform-tools

ABI="arm64-v8a"
STL="c++_static"
SHARED_LIB="ON"
PROFILING="OFF"
CLEAN=""
PUSH_MODEL=""
DEVICE_TYPE=""
MODEL_TYPE=TNN
USE_NCNN_MODEL=0
KERNEL_TUNE="-et"
THREAD_NUM=1
ADB=adb

WORK_DIR=`pwd`
BENCHMARK_MODEL_DIR=$WORK_DIR/../benchmark-model
BUILD_DIR=build
ANDROID_DIR=/data/local/tmp/tnn-benchmark
ANDROID_DATA_DIR=$ANDROID_DIR/benchmark-model
OUTPUT_LOG_FILE=benchmark_models_result.txt
LOOP_COUNT=16
WARM_UP_COUNT=5

benchmark_model_list=(
#test.tnnproto \
)

function usage() {
    echo "usage: ./benchmark_models.sh  [-32] [-c] [-b] [-f] [-d] <device-id> [-t] <CPU/GPU>"
    echo "options:"
    echo "        -32   Build 32 bit."
    echo "        -c    Clean up build folders."
    echo "        -b    build targets only"
    echo "        -f    build profiling targets "
    echo "        -d    run with specified device"
    echo "        -t    CPU/GPU/HUAWEI_NPU specify the platform to run"
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
    if [ "$DEVICE_TYPE" = "HUAWEI_NPU"  ]; then
      echo "NPU Enable"
      # set c++ shared
      STL="c++_shared"
      HUAWEI_NPU_ENABLE="ON"
      #start to cp
      if [ ! -d ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/ ]; then
           mkdir -p ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/
      fi
      mkdir -p ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/armeabi-v7a
      mkdir -p ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/arm64-v8a
      cp $ANDROID_NDK/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a/libc++_shared.so  ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/armeabi-v7a/
      cp $ANDROID_NDK/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/arm64-v8a/
    fi
    mkdir -p build
    cd $BUILD_DIR
    cmake ../../.. \
          -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DANDROID_ABI="${ABI}" \
          -DANDROID_STL=${STL}\
          -DANDROID_NATIVE_API_LEVEL=android-14  \
          -DANDROID_TOOLCHAIN=clang \
          -DTNN_ARM_ENABLE:BOOL=ON \
          -DTNN_OPENCL_ENABLE:BOOL=ON \
          -DTNN_HUAWEI_NPU_ENABLE:BOOL=${HUAWEI_NPU_ENABLE} \
          -DTNN_OPENMP_ENABLE:BOOL=ON \
          -DTNN_TEST_ENABLE:BOOL=ON \
          -DTNN_BENCHMARK_MODE:BOOL=ON \
          -DTNN_PROFILER_ENABLE:BOOL=${PROFILING} \
          -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB \
          -DBUILD_FOR_ANDROID_COMMAND=true
    make -j4
}

function bench_android() {
    build_android_bench

    if [ $? != 0 ];then
        exit_with_msg "build failed"
    fi

    if [ "" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi

    $ADB shell "mkdir -p $ANDROID_DIR"
    find . -name "*.so" | while read solib; do
        $ADB push $solib  $ANDROID_DIR
    done
    $ADB push test/TNNTest $ANDROID_DIR/TNNTest
    $ADB shell chmod 0777 $ANDROID_DIR/TNNTest

    $ADB shell "mkdir -p $ANDROID_DIR/benchmark-model"
    $ADB push ${BENCHMARK_MODEL_DIR} $ANDROID_DIR

    cd ${BENCHMARK_MODEL_DIR}
    $ADB shell "getprop ro.product.model > ${ANDROID_DIR}/$OUTPUT_LOG_FILE"

    if [ ${#benchmark_model_list[*]} == 0 ];then
        benchmark_model_list=`ls *.tnnproto`
    fi

    if [ "$DEVICE_TYPE" != "GPU" ] && [ "$DEVICE_TYPE" != "CPU" ] && [ "$DEVICE_TYPE" != "HUAWEI_NPU" ]; then
        DEVICE_TYPE=""
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "CPU" ];then
        device=ARM
        $ADB shell "echo '\nbenchmark device: ${device} \n' >> ${ANDROID_DIR}/$OUTPUT_LOG_FILE"

        for benchmark_model in ${benchmark_model_list[*]}
        do
            $ADB shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=. ./TNNTest -th ${THREAD_NUM} ${KERNEL_TUNE} -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mt ${MODEL_TYPE} -mp ${ANDROID_DATA_DIR}/${benchmark_model}  >> $OUTPUT_LOG_FILE"
        done
    fi

    if [ "ON" == $PROFILING ]; then
        WARM_UP_COUNT=5
        LOOP_COUNT=5
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "GPU" ];then
        device=OPENCL
        $ADB shell "echo '\nbenchmark device: ${device} \n' >> ${ANDROID_DIR}/$OUTPUT_LOG_FILE"
        for benchmark_model in ${benchmark_model_list[*]}
        do
            $ADB shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=. ./TNNTest -th ${THREAD_NUM} ${KERNEL_TUNE} -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mt ${MODEL_TYPE} -mp ${ANDROID_DATA_DIR}/${benchmark_model}  >> $OUTPUT_LOG_FILE"
        done
    fi

    if [ "$DEVICE_TYPE" = "HUAWEI_NPU" ];then
        echo "Run Huawei Npu"
        device=HUAWEI_NPU
	    $ADB push ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/${ABI}/* $ANDROID_DIR/
        $ADB push ${WORK_DIR}/../../third_party/huawei_npu/hiai_ddk_latest/${ABI}/* $ANDROID_DIR/
        $ADB shell "echo '\nbenchmark device: ${device} \n' >> ${ANDROID_DIR}/$OUTPUT_LOG_FILE"
        for benchmark_model in ${benchmark_model_list[*]}
        do
            $ADB shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=. ./TNNTest -th ${THREAD_NUM} ${KERNEL_TUNE} -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -nt ${device} -mt ${MODEL_TYPE} -mp ${ANDROID_DATA_DIR}/${benchmark_model}  >> $OUTPUT_LOG_FILE"
        done
    fi

    $ADB shell "echo '' >> $ANDROID_DIR/$OUTPUT_LOG_FILE"
    $ADB shell "date  >> $ANDROID_DIR/$OUTPUT_LOG_FILE"

    $ADB pull $ANDROID_DIR/$OUTPUT_LOG_FILE ${WORK_DIR}/$OUTPUT_LOG_FILE
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
        -b)
            shift
            BUILD_ONLY="-b"
            ;;
        -f)
            shift
            PROFILING="ON"
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
        -n)
            shift
            MODEL_TYPE=NCNN
            ;;
        -th)
            shift
            THREAD_NUM=$1
            shift
            ;;
        *)
            usage
            exit 1
    esac
done

bench_android
