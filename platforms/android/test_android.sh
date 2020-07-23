#!/bin/bash

ANDROID_DIR=/data/local/tmp/

TEST_PROTO_PATH=

DEVICE="ARM"
WARM_UP_COUNT=0
ITERATOR_COUNT=1
NEED_REBUILD=true
NEED_PUSH=true
INPUT_PATH=

if $NEED_REBUILD
then
    ../../scripts/build_android.sh
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

if [ -n "$INPUT_PATH" ]
then
    adb shell "cd $ANDROID_DIR; LD_LIBRARY_PATH=$ANDROID_DIR ./TNNTest -dt=${DEVICE} -mp=./test.tnnproto -ip=input.txt -op=${DEVICE}_output.data -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT >> $ANDROID_DIR/test.log"
else
    adb shell "cd $ANDROID_DIR; LD_LIBRARY_PATH=$ANDROID_DIR ./TNNTest -dt=${DEVICE} -mp=./test.tnnproto -op=${DEVICE}_output.data -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT >> $ANDROID_DIR/test.log"
fi

adb pull $ANDROID_DIR/${DEVICE}_output.data ${DEVICE}_output.data
adb pull $ANDROID_DIR/test.log ${DEVICE}_test.log
