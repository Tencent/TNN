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
INPUT_FILE_NAME=rpn_in_0_n1_c3_h320_w320.txt

function usage() {
    echo "-64\tBuild 64bit."
    echo "-c\tClean up build folders."
    echo "-p\tPush models to device"
    echo "-b\tbuild targets only"
}
function die() {
    echo $1
    exit 1
}

function clean_build() {
    echo $1 | grep "$BUILD_DIR\b" > /dev/null
    if [[ "$?" != "0" ]]; then
        die "Warnning: $1 seems not to be a BUILD folder."
    fi
    rm -rf $1
    mkdir $1
}

function build_android() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p build
    cd $BUILD_DIR
    cmake ../../../ \
          -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DANDROID_ABI="${ABI}" \
          -DANDROID_STL=c++_static \
          -DANDROID_NATIVE_API_LEVEL=android-24  \
          -DANDROID_TOOLCHAIN=clang \
          -DANDROID_TEST_ENABLE=1 \
          -DTNN_NPU_ENABLE:BOOL=$NPU \
          -DBUILD_FOR_ANDROID_COMMAND=true
    make -j4
}

function push_hiai_lib() {
    adb shell "mkdir -p $ANDROID_DIR/$ABI/lib"
    adb push $WORK_DIR/../../source/device/npu/thirdparty/hiai_ddk_200/lib64/* $ANDROID_DIR/$ABI/lib
}

function run_android() {
    build_android
    if [ "" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi
    mkdir -p $DUMP_DIR

    adb shell "mkdir -p $ANDROID_DIR"
    find . -name "*.so" | while read solib; do
        adb push $solib  $ANDROID_DIR
    done
    adb push AndroidTest $ANDROID_DIR
    adb shell chmod 0777 $ANDROID_DIR/AndroidTest

    if [ "" != "$PUSH_MODEL" ]; then
        adb shell "rm -r $ANDROID_DATA_DIR"
        adb shell "mkdir -p $ANDROID_DATA_DIR"
        adb push $MODEL_DIR/* $ANDROID_DATA_DIR
        push_hiai_lib
    fi
    adb shell "cat /proc/cpuinfo > $ANDROID_DIR/test_log.txt"
    adb shell "echo >> $ANDROID_DIR/test_log.txt"
    adb shell "mkdir -p $ANDROID_DIR/dump_data"
    adb shell "cd $ANDROID_DIR ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ANDROID_DIR/$ABI/lib:$ANDROID_DIR; ./AndroidTest $ANDROID_DATA_DIR/test.om $ANDROID_DATA_DIR/$INPUT_FILE_NAME  >> $ANDROID_DIR/test_log.txt"
    adb pull $ANDROID_DIR/test_log.txt $DUMP_DIR
    adb pull $ANDROID_DIR/dump_data.txt $DUMP_DIR
    adb pull $ANDROID_DIR/dump_data.bin $DUMP_DIR
    adb pull $ANDROID_DIR/dump_data $DUMP_DIR
}

while [ "$1" != "" ]; do
    case $1 in
        -64)
            shift
            ABI="arm64-v8a"
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
