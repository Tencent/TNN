#!/bin/bash
ABI="armeabi-v7a with NEON"
STL="c++_static"
SHARED_LIB="ON"
CLEAN=""
PUSH_MODEL=""
BUILD_ONLY=""
PROFILING="OFF"
SHARING_MEM_WITH_OPENGL=0
WARM_UP_COUNT=30
ITERATOR_COUNT=100
ADB=adb

WORK_DIR=`pwd`
BUILD_DIR=build
MODEL_DIR=$WORK_DIR/models
ANDROID_DIR=/data/local/tmp/ocl_test
ANDROID_DATA_DIR=$ANDROID_DIR/data
DUMP_DIR=$WORK_DIR/dump_data
OUTPUT_LOG_FILE=test_log.txt

benchmark_model_list=(
test.rapidproto \
)

function usage() {
    echo "-64\tBuild 64bit."
    echo "-c\tClean up build folders."
    echo "-p\tPush models to device"
    echo "-b\tbuild targets only"
    echo "-f\tbuild profiling targets "
    echo "-d\trun with specified device"
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
          -DTNN_ARM_ENABLE:BOOL="ON" \
          -DTNN_OPENMP_ENABLE:BOOL="ON" \
          -DTNN_OPENCL_ENABLE:BOOL="ON" \
          -DTNN_TEST_ENABLE:BOOL="ON"  \
          -DTNN_BENCHMARK_MODE:BOOL="ON" \
          -DTNN_PROFILER_ENABLE:BOOL=${PROFILING} \
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

    mkdir -p $DUMP_DIR
    $ADB shell "mkdir -p $ANDROID_DIR"
    find . -name "*.so" | while read solib; do
        $ADB push $solib  $ANDROID_DIR
    done
    $ADB push test/TNNTest $ANDROID_DIR
    $ADB shell chmod 0777 $ANDROID_DIR/TNNTest

    if [ "" != "$PUSH_MODEL" ]; then
        $ADB shell "rm -r $ANDROID_DATA_DIR"
        $ADB shell "mkdir -p $ANDROID_DATA_DIR"
        $ADB push $MODEL_DIR/* $ANDROID_DATA_DIR
    fi
    $ADB shell "echo > $ANDROID_DIR/$OUTPUT_LOG_FILE"

    $ADB shell "echo 'device: ARM CPU'  >> $ANDROID_DIR/$OUTPUT_LOG_FILE"
    for benchmark_model in ${benchmark_model_list[*]}
    do
        # -dl  0,1,2,3 small core   4,5 big core
        $ADB shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=$ANDROID_DIR  ./TNNTest -dt=ARM  -wc=${WARM_UP_COUNT} -ic=${ITERATOR_COUNT} -mp=${ANDROID_DATA_DIR}/${benchmark_model}  >> $OUTPUT_LOG_FILE"
    done

    if [ "ON" == $PROFILING ]; then
        WARM_UP_COUNT=10
        ITERATOR_COUNT=1
    fi

    $ADB shell "echo 'device: OPENCL'  >> $ANDROID_DIR/$OUTPUT_LOG_FILE"
    for benchmark_model in ${benchmark_model_list[*]}
    do
        $ADB shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=$ANDROID_DIR  ./TNNTest -dt=OPENCL -wc=${WARM_UP_COUNT} -ic=${ITERATOR_COUNT} -mp=${ANDROID_DATA_DIR}/${benchmark_model}  >> $OUTPUT_LOG_FILE"
    done

    $ADB shell "getprop ro.product.model  | sed 's/[ ][ ]*/_/g' >> $ANDROID_DIR/$OUTPUT_LOG_FILE"
    $ADB shell "date  >> $ANDROID_DIR/$OUTPUT_LOG_FILE"

    $ADB pull $ANDROID_DIR/$OUTPUT_LOG_FILE $DUMP_DIR
    cat $DUMP_DIR/$OUTPUT_LOG_FILE
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
