#!/bin/bash
ABI="armeabi-v7a"
TARGET_ARCH=arm-android-clang6.0
DSP="ON"
FORWARD_TYPE=0
CLEAN=""
PUSH_MODEL=""
BUILD_ONLY=""

WORK_DIR=`pwd`
BUILD_DIR=build
MODEL_DIR=$WORK_DIR/dlc
ANDROID_DIR=/data/local/tmp/dsp_test
ANDROID_DATA_DIR=$ANDROID_DIR/data
DUMP_DIR=$WORK_DIR/dump_data_dsp
#INPUT_FILE_NAME=rpn_in_0_n1_c3_h320_w320.txt
INPUT_FILE_NAME=input_320x320_nhwc_fp32.bin
#INPUT_FILE_NAME=rpn_in_input_array_n1_c3_h256_w192.txt

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
          -DANDROID_NATIVE_API_LEVEL=android-14  \
          -DANDROID_TOOLCHAIN=clang \
          -DANDROID_TEST_ENABLE=1 \
          -DTNN_DSP_ENABLE:BOOL=$DSP \
          -DBUILD_FOR_ANDROID_COMMAND=true
    make -j4
}

function push_snpe_lib() {
    adb shell "mkdir -p $ANDROID_DIR/$TARGET_ARCH/lib"
    adb shell "mkdir -p $ANDROID_DIR/dsp/lib"
    adb push $SNPE_ROOT/lib/$TARGET_ARCH/ $ANDROID_DIR/$TARGET_ARCH/lib
    adb push $SNPE_ROOT/lib/dsp/ $ANDROID_DIR/dsp/lib
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
        push_snpe_lib
    fi
    adb shell "cat /proc/cpuinfo > $ANDROID_DIR/test_log.txt"
    adb shell "echo >> $ANDROID_DIR/test_log.txt"
    adb shell "mkdir -p $ANDROID_DIR/dump_data"
    adb shell "cd $ANDROID_DIR ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ANDROID_DIR/$TARGET_ARCH/lib:$ANDROID_DIR; export ADSP_LIBRARY_PATH=\"$ANDROID_DIR/dsp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp\"; ./AndroidTest $ANDROID_DATA_DIR/test.dlc $ANDROID_DATA_DIR/$INPUT_FILE_NAME  >> $ANDROID_DIR/test_log.txt"
    adb pull $ANDROID_DIR/test_log.txt $DUMP_DIR
    adb pull $ANDROID_DIR/dump_data.txt $DUMP_DIR
    adb pull $ANDROID_DIR/dump_data.bin $DUMP_DIR
    #adb pull $ANDROID_DIR/result_rgb.bin $DUMP_DIR
    #adb pull $ANDROID_DIR/result_alpha.bin $DUMP_DIR
    adb pull $ANDROID_DIR/dump_data $DUMP_DIR
}

while [ "$1" != "" ]; do
    case $1 in
        -64)
            shift
            ABI="arm64-v8a"
            TARGET_ARCH=aarch64-android-clang6.0
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
