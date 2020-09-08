#!/bin/bash
ABI="armeabi-v7a"
CLEAN=""
PUSH_MODEL=""
BUILD_ONLY=""
BUILD="32"
WORK_DIR=`pwd`
MODEL_DIR=$WORK_DIR//
ANDROID_DIR=/data/local/tmp/npu_test
ANDROID_DATA_DIR=$ANDROID_DIR/data
DUMP_DIR=$WORK_DIR/dump_data_npu

MODEL_TYPE=tnn
MODEL_NAME=
#INPUT_FILE_NAME=input_128.txt

function usage() {
    echo "-64\tBuild 64bit."
    echo "-p\tPush models to device"
    echo "-b\tbuild targets only"
}
function die() {
    echo $1
    exit 1
}


function run_android() {
    BUILD_DIR=../../scripts
    cd $BUILD_DIR
    export NPU="ON"
    if [ "$CLEAN" == "" ]; then
        ./build_android.sh -ic
    else
        ./build_android.sh
    fi

    cd ../platforms/android

    if [ "" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi
    mkdir -p $DUMP_DIR

    adb shell "mkdir -p $ANDROID_DIR"
    adb push  $BUILD_DIR/build${BUILD}/test/TNNTest $ANDROID_DIR
    adb shell chmod 0777 $ANDROID_DIR/TNNTest

    if [ "" != "$PUSH_MODEL" ]; then
        adb shell "rm -r $ANDROID_DATA_DIR"
        adb shell "mkdir -p $ANDROID_DATA_DIR"
        adb push $MODEL_DIR/* $ANDROID_DATA_DIR
        adb shell "mkdir -p $ANDROID_DIR/$ABI/lib"
        adb push $WORK_DIR/../../third_party/npu/cpp_lib/$ABI/* $ANDROID_DIR/$ABI/lib
        adb push ${BUILD_DIR}/release/$ABI/* $ANDROID_DIR/$ABI/lib
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
            BUILD=64
            ;;
        -c)
            shift
            CLEAN="-c"
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
