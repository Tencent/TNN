#!/bin/bash

ABI="arm64-v8a"
ADB=adb
ANDROID_DIR=/data/local/tmp/tnn-test

TEST_PROTO_PATH=
#DEVIVE: ARM/OPENCL/HUAWEI_NPU
DEVICE="ARM"
WARM_UP_COUNT=0
ITERATOR_COUNT=1
NEED_CLEAN=false
NEED_PUSH=true
INPUT_PATH=

WORK_DIR=`pwd`

function usage() {
    echo "usage: ./test_android.sh  [-32] [-c] [-d] <device-id> [-t] <CPU/GPU> -m <tnnproto file path> -i <input file path>"
    echo "options:"
    echo "        -32   Build 32 bit."
    echo "        -c    Clean up build folders."
    echo "        -d    run with specified device"
    echo "        -t    ARM/OPENCL/HUAWEI_NPU specify the platform to run (default: ARM)"
    echo "        -m    tnnproto"
    echo "        -i    input file"
}

function android_test() {

    if [ "$DEVICE" == "HUAWEI_NPU" ]
    then
        export HUAWEI_NPU="ON"
    else
        export HUAWEI_NPU="OFF"
    fi
    if $NEED_CLEAN
    then
        rm -r build32 build64
    fi
    ../../scripts/build_android.sh -ic

    if $NEED_PUSH
    then
        $ADB shell "mkdir -p $ANDROID_DIR"
        if [ "$ABI" == "arm64-v8a" ]; then
            $ADB push build64/libTNN.so ${ANDROID_DIR}/libTNN.so
            $ADB push build64/test/TNNTest ${ANDROID_DIR}/TNNTest
        else
            $ADB push build32/libTNN.so ${ANDROID_DIR}/libTNN.so
            $ADB push build32/test/TNNTest ${ANDROID_DIR}/TNNTest
        fi
        if [ -z "$TEST_PROTO_PATH" ]
        then
            TEST_PROTO_PATH=../../model/SqueezeNet/squeezenet_v1.1.tnnproto
        fi
        if [ -n "$INPUT_PATH" ]
        then
            echo "input path"
            $ADB push ${INPUT_PATH} ${ANDROID_DIR}/input.txt
        fi
        TEST_MODEL_PATH=${TEST_PROTO_PATH/proto/model}
        $ADB push ${TEST_PROTO_PATH} ${ANDROID_DIR}/test.tnnproto
        $ADB push ${TEST_MODEL_PATH} ${ANDROID_DIR}/test.tnnmodel
    fi

    $ADB shell "echo "${DEVICE}" > $ANDROID_DIR/test.log"
    if [ "$DEVICE" == "HUAWEI_NPU" ]
    then
        echo "Run Huawei Npu"
        $ADB shell "mkdir -p $ANDROID_DIR/lib"
        $ADB push $WORK_DIR/../../third_party/huawei_npu/cpp_lib/$ABI/* $ANDROID_DIR/lib
        $ADB push $WORK_DIR/../../third_party/huawei_npu/hiai_ddk_latest/$ABI/* $ANDROID_DIR/lib
        if [ -n "$INPUT_PATH" ]
        then
          $ADB shell "cd $ANDROID_DIR; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ANDROID_DIR}/lib:$ANDROID_DIR; ./TNNTest -dt=${DEVICE} -nt=HUAWEI_NPU -mp=./test.tnnproto -ip=input.txt -op=${DEVICE}_output.data -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT"
        else
          $ADB shell "cd $ANDROID_DIR; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ANDROID_DIR}/lib:$ANDROID_DIR; ./TNNTest -dt=${DEVICE} -nt=HUAWEI_NPU -mp=./test.tnnproto -op=${DEVICE}_output.data -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT"
        fi
    else
        if [ -n "$INPUT_PATH" ]
        then
          $ADB shell "cd $ANDROID_DIR; LD_LIBRARY_PATH=$ANDROID_DIR ./TNNTest -dt=${DEVICE} -mp=./test.tnnproto -ip=input.txt -op=${DEVICE}_output.data -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT"
        else
          $ADB shell "cd $ANDROID_DIR; LD_LIBRARY_PATH=$ANDROID_DIR ./TNNTest -dt=${DEVICE} -mp=./test.tnnproto -op=${DEVICE}_output.data -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT"
        fi
    fi
    $ADB shell "cd ${ANDROID_DIR}; logcat -d | grep \"TNN Benchmark time cost\" | grep ${DEVICE} | grep -w \"test.tnnproto\" | tail -n 1 >> $ANDROID_DIR/test.log"

    $ADB pull $ANDROID_DIR/${DEVICE}_output.data ${DEVICE}_output.data
    $ADB pull $ANDROID_DIR/test.log ${DEVICE}_test.log
}

while [ "$1" != "" ]; do
    case $1 in
        -32)
            shift
            ABI="armeabi-v7a"
            ;;
        -c)
            shift
            NEED_CLEAN=true
            ;;
        -d)
            shift
            ADB="adb -s $1"
            shift
            ;;
        -t)
            shift
            DEVICE="$1"
            shift
            ;;
        -m)
            shift
            TEST_PROTO_PATH="$1"
            shift
            ;;
        -i)
            shift
            INPUT_PATH="$1"
            shift
            ;;
        *)
            usage
            exit 1
    esac
done

android_test
