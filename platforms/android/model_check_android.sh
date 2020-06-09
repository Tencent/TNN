#!/bin/bash
ABI="armeabi-v7a"
#STL="gnustl_static"
STL="c++_static"
CLEAN=""
PUSH_MODEL=""
BUILD_ONLY=""

WORK_DIR=`pwd`
BUILD_DIR=build
MODEL_DIR=$WORK_DIR/models
ANDROID_DIR=/data/local/tmp/ocl_test
ANDROID_DATA_DIR=$ANDROID_DIR/data
DUMP_DIR=$WORK_DIR/dump_data_ocl
ADB=adb

check_model_list=(
"test.tnnproto" \
                      )

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
          -DANDROID_STL=${STL}\
          -DANDROID_NATIVE_API_LEVEL=android-14  \
          -DANDROID_TOOLCHAIN=clang \
          -DTNN_CPU_ENABLE:BOOL="ON"  \
          -DTNN_OPENCL_ENABLE:BOOL="ON" \
          -DTNN_ARM_ENABLE:BOOL="ON" \
          -DTNN_MODEL_CHECK_ENABLE:BOOL="ON" \
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

    mkdir -p $DUMP_DIR
    $ADB shell "mkdir -p $ANDROID_DIR"
    find . -name "*.so" | while read solib; do
        $ADB push $solib  $ANDROID_DIR
    done
    $ADB push model_check $ANDROID_DIR
    $ADB shell chmod 0777 $ANDROID_DIR/model_check

    if [ "" != "$PUSH_MODEL" ]; then
        $ADB shell "rm -r $ANDROID_DATA_DIR"
        $ADB shell "mkdir -p $ANDROID_DATA_DIR"
        $ADB push $MODEL_DIR/* $ANDROID_DATA_DIR
    fi
    $ADB shell "echo > $ANDROID_DIR/test_log.txt"
    $ADB shell "mkdir -p $ANDROID_DIR/dump_data"

    for check_model in ${check_model_list[*]}
    do
        model_name=${check_model%.*}
        #$ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./model_check -d ARM -p $ANDROID_DATA_DIR/"$model_name".tnnproto -m $ANDROID_DATA_DIR/"$model_name".tnnmodel >> $ANDROID_DIR/test_log.txt"
        $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./model_check -d OPENCL -p $ANDROID_DATA_DIR/"$model_name".tnnproto -m $ANDROID_DATA_DIR/"$model_name".tnnmodel >> $ANDROID_DIR/test_log.txt"
    done
    $ADB pull $ANDROID_DIR/test_log.txt $DUMP_DIR
    $ADB pull $ANDROID_DIR/dump_data $DUMP_DIR
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

run_android
