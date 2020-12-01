#!/bin/bash

ANDROID_DIR=/data/local/tmp/

TEST_PROTO_PATH=
#DEVIVE: ARM/OPENCL/HUAWEI_NPU
DEVICE="ARM"
WARM_UP_COUNT=0
ITERATOR_COUNT=1
NEED_REBUILD=true
NEED_PUSH=true
INPUT_PATH=

WORK_DIR=`pwd`
ABI="armeabi-v7a"
if [ "$DEVICE" == "HUAWEI_NPU" ]
then
    export HUAWEI_NPU="ON"
else 
    export HUAWEI_NPU="OFF"
fi
if $NEED_REBUILD
then
    ../../scripts/build_android.sh -ic
fi

if $NEED_PUSH
then
    adb push build32/libTNN.so ${ANDROID_DIR}/libTNN.so
    adb push build32/test/TNNTest ${ANDROID_DIR}/TNNTest
    if [ -z "$TEST_PROTO_PATH" ]
    then
        TEST_PROTO_PATH=../../model/SqueezeNet/squeezenet_v1.1.tnnproto
    fi
    if [ -n "$INPUT_PATH" ]
    then
        echo "input path"
        adb push ${INPUT_PATH} ${ANDROID_DIR}/input.txt
    fi
    TEST_MODEL_PATH=${TEST_PROTO_PATH/proto/model}
    adb push ${TEST_PROTO_PATH} ${ANDROID_DIR}/test.tnnproto
    adb push ${TEST_MODEL_PATH} ${ANDROID_DIR}/test.tnnmodel
fi

adb shell "echo "${DEVICE}" > $ANDROID_DIR/test.log"
if [ "$DEVICE" == "HUAWEI_NPU" ]
then
    echo "Run Huawei Npu"
    adb shell "mkdir -p $ANDROID_DIR/lib"
    adb push $WORK_DIR/../../third_party/huawei_npu/cpp_lib/$ABI/* $ANDROID_DIR/lib
    adb push $WORK_DIR/../../third_party/huawei_npu/hiai_ddk_latest/$ABI/* $ANDROID_DIR/lib
    if [ -n "$INPUT_PATH" ]
    then
      adb shell "cd $ANDROID_DIR; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ANDROID_DIR}/lib:$ANDROID_DIR; ./TNNTest -dt=${DEVICE} -nt=HUAWEI_NPU -mp=./test.tnnproto -ip=input.txt -op=${DEVICE}_output.data -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT >> $ANDROID_DIR/test.log"
    else
      adb shell "cd $ANDROID_DIR; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ANDROID_DIR}/lib:$ANDROID_DIR; ./TNNTest -dt=${DEVICE} -nt=HUAWEI_NPU -mp=./test.tnnproto -op=${DEVICE}_output.data -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT >> $ANDROID_DIR/test.log"
    fi
else
    if [ -n "$INPUT_PATH" ]
    then
      adb shell "cd $ANDROID_DIR; LD_LIBRARY_PATH=$ANDROID_DIR ./TNNTest -dt=${DEVICE} -mp=./test.tnnproto -ip=input.txt -op=${DEVICE}_output.data -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT >> $ANDROID_DIR/test.log"
    else
      adb shell "cd $ANDROID_DIR; LD_LIBRARY_PATH=$ANDROID_DIR ./TNNTest -dt=${DEVICE} -mp=./test.tnnproto -op=${DEVICE}_output.data -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT >> $ANDROID_DIR/test.log"
    fi
fi

adb pull $ANDROID_DIR/${DEVICE}_output.data ${DEVICE}_output.data
adb pull $ANDROID_DIR/test.log ${DEVICE}_test.log
