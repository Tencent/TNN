#!/bin/bash
ABI="armeabi-v7a"
#STL="gnustl_static"
STL="c++_static"
OPENCL="ON"
SHARED_LIB="ON"
CLEAN=""
PUSH_MODEL=""
BUILD_ONLY=""
PROFILING="OFF"
SHARING_MEM_WITH_OPENGL=0
WARM_UP_COUNT=10
ITERATOR_COUNT=100

WORK_DIR=`pwd`
BUILD_DIR=build
MODEL_DIR=$WORK_DIR/models
ANDROID_DIR=/data/local/tmp/ocl_test
ANDROID_DATA_DIR=$ANDROID_DIR/data
DUMP_DIR=$WORK_DIR/dump_data_ocl
INPUT_FILE_NAME=rpn_in_0_n1_c3_h320_w320.txt
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
          -DSHARING_MEM_WITH_OPENGL=${SHARING_MEM_WITH_OPENGL} \
          -DANDROID_ABI="${ABI}" \
          -DANDROID_STL=${STL}\
          -DANDROID_NATIVE_API_LEVEL=android-14  \
          -DANDROID_TOOLCHAIN=clang \
          -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
          -DTNN_PROFILER_ENABLE:BOOL=${PROFILING} \
          -DTNN_TEST_ENABLE:BOOL="ON"  \
          -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB \
          -DBUILD_FOR_ANDROID_COMMAND=true
    make -j4
}

function run_android() {
    build_android
    if [ $? != 0 ];then
        echo "build falied"
        exit 0
    fi

    if [ "" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi

    if [ "ON" == $PROFILING ]; then
        WARM_UP_COUNT=0
        ITERATOR_COUNT=1
    fi

    mkdir -p $DUMP_DIR
    adb shell "mkdir -p $ANDROID_DIR"
    find . -name "*.so" | while read solib; do
        adb push $solib  $ANDROID_DIR
    done
    adb push test/TNNTest $ANDROID_DIR
    adb shell chmod 0777 $ANDROID_DIR/TNNTest

    if [ "" != "$PUSH_MODEL" ]; then
        adb shell "rm -r $ANDROID_DATA_DIR"
        adb shell "mkdir -p $ANDROID_DATA_DIR"
        adb push $MODEL_DIR/* $ANDROID_DATA_DIR
    fi
    adb shell "echo > $ANDROID_DIR/test_log.txt"
    adb shell "mkdir -p $ANDROID_DIR/dump_data"
    adb shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./TNNTest -dt=OPENCL -mp=$ANDROID_DATA_DIR/test.tnnproto -ip=$ANDROID_DATA_DIR/$INPUT_FILE_NAME -op=dump_data.txt -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT >> $ANDROID_DIR/test_log.txt"
    adb pull $ANDROID_DIR/test_log.txt $DUMP_DIR
    adb pull $ANDROID_DIR/dump_data.txt $DUMP_DIR
    #adb pull $ANDROID_DIR/dump_data.bin $DUMP_DIR
    #adb pull $ANDROID_DIR/result_rgb.bin $DUMP_DIR
    #adb pull $ANDROID_DIR/result_alpha.bin $DUMP_DIR
    #adb pull $ANDROID_DIR/dump_data $DUMP_DIR

    cat $DUMP_DIR/test_log.txt
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
        -f)
            shift
            PROFILING="ON"
            ;;
        *)
            usage
            exit 1
    esac
done

run_android
