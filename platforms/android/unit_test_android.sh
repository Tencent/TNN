#!/bin/bash
ABI="armeabi-v7a"
OPENCL="ON"
CLEAN=""
BUILD_ONLY=""

WORK_DIR=`pwd`
BUILD_DIR=build
FILTER=""
ANDROID_DIR=/data/local/tmp/ocl_test
ANDROID_DATA_DIR=$ANDROID_DIR/data
DUMP_DIR=$WORK_DIR/dump_data_ocl
ADB=adb

function usage() {
    echo "-64\tBuild 64bit."
    echo "-c\tClean up build folders."
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
          -DTNN_TEST_ENABLE:BOOL="ON"  \
          -DTNN_UNIT_TEST_ENABLE:BOOL="ON"  \
          -DTNN_ARM_ENABLE:BOOL="ON" \
          -DTNN_OPENCL_ENABLE:BOOL=$OPENCL \
          -DBUILD_FOR_ANDROID_COMMAND=true
    make -j4
}

function run() {
    build_android
    if [ $? != 0 ]; then
        echo "build falied!"
        exit 0
    fi

    mkdir -p $DUMP_DIR

    $ADB shell "mkdir -p $ANDROID_DIR"
    find . -name "*.so" | while read solib; do
        $ADB push $solib  $ANDROID_DIR
    done
    $ADB push  test/unit_test/unit_test  $ANDROID_DIR
    $ADB shell chmod 0777 $ANDROID_DIR/unit_test

    $ADB shell "mkdir -p $ANDROID_DIR/dump_data"

    if [ "" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi

    $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./unit_test -dt ARM  --gtest_filter=\"*${FILTER}*\" > $ANDROID_DIR/test_log.txt"
    $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./unit_test -dt OPENCL  --gtest_filter=\"*${FILTER}*\" > $ANDROID_DIR/test_log.txt"
    $ADB pull $ANDROID_DIR/test_log.txt $DUMP_DIR
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
        -b)
            shift
            BUILD_ONLY="-b"
            ;;
        -f)
            shift
            FILTER=$1
            shift
            ;;
        -d)
            shift
            ADB="adb -s $1"
            shift
            ;;
        *)
            usage
            exit 1
    esac
done

run
