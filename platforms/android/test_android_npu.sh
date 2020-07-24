#!/bin/bash
ABI="armeabi-v7a"
NPU="ON"
CLEAN=""
PUSH_MODEL=""
BUILD_ONLY=""

WORK_DIR=`pwd`
BUILD_DIR=build
MODEL_DIR=$WORK_DIR/npu_model
ANDROID_DIR=/data/local/tmp/npu_test
ANDROID_DATA_DIR=$ANDROID_DIR/data
DUMP_DIR=$WORK_DIR/dump_data_npu

MODEL_TYPE=tnn
MODEL_NAME=deconv+reducesum.tnnproto
INPUT_FILE_NAME=input_128.txt

function usage() {
    echo "-64\tBuild 64bit."
    echo "-p\tPush models to device"
    echo "-b\tbuild targets only"
}
function die() {
    echo $1
    exit 1
}

function push_tnn() {
    adb shell "rm -rf $ANDROID_DIR/$ABI/lib"
    adb shell "mkdir -p $ANDROID_DIR/$ABI/lib"
    adb push $WORK_DIR/../../third_party/npu/hiai_ddk_latest/$ABI/* $ANDROID_DIR/$ABI/lib
		adb push $WORK_DIR/../../third_party/npu/cpp_lib/$ABI/* $ANDROID_DIR/$ABI/lib
    adb push ./build${1}/libTNN.so  $ANDROID_DIR
    adb push  $WORK_DIR/build${1}/test/TNNTest $ANDROID_DIR
    adb shell chmod 0777 $ANDROID_DIR/TNNTest
}
function run_android() {
    ../../scripts/build_android.sh
    if [ "" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi
    mkdir -p $DUMP_DIR

    adb shell "mkdir -p $ANDROID_DIR"
    
    if [ $ABI == "arm64-v8a" ]; then
        push_tnn 64
    else
    	  push_tnn 32
    fi

    if [ "" != "$PUSH_MODEL" ]; then
        adb shell "rm -r $ANDROID_DATA_DIR"
        adb shell "mkdir -p $ANDROID_DATA_DIR"
        adb push $MODEL_DIR/* $ANDROID_DATA_DIR
    fi
    adb shell "echo > $ANDROID_DIR/test_log.txt"
    adb shell "mkdir -p $ANDROID_DIR/dump_data"
    echo "cd $ANDROID_DIR ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ANDROID_DIR/$ABI/lib:$ANDROID_DIR; ./TNNTest -mt $MODEL_TYPE -nt NPU -mp $ANDROID_DATA_DIR/$MODEL_NAME -dt NPU -ip $ANDROID_DATA_DIR/$INPUT_FILE_NAME -op $ANDROID_DIR/dump_data.txt   >> $ANDROID_DIR/test_log.txt"
    adb shell "cd $ANDROID_DIR ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ANDROID_DIR/$ABI/lib:$ANDROID_DIR; ./TNNTest -mt $MODEL_TYPE -nt NPU -mp $ANDROID_DATA_DIR/$MODEL_NAME -dt NPU -ip $ANDROID_DATA_DIR/$INPUT_FILE_NAME -op $ANDROID_DIR/dump_data.txt   >> $ANDROID_DIR/test_log.txt"
    adb pull $ANDROID_DIR/test_log.txt $DUMP_DIR
    adb pull $ANDROID_DIR/dump_data.txt $DUMP_DIR
    cat $DUMP_DIR/test_log.txt
}

while [ "$1" != "" ]; do
    case $1 in
        -64)
            shift
            ABI="arm64-v8a"
            ;;
        -p)
            shift
            PUSH_MODEL="-p"
            ;;
        -b)
            shift
            BUILD_ONLY="-b"
            ;;
        *)
            usage
            exit 1
    esac
done

run_android
