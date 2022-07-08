#!/bin/bash
ABI="arm64-v8a"
STL="c++_static"
CLEAN=""
PUSH_MODEL=""
BUILD_ONLY=""

WORK_DIR=`pwd`
BUILD_DIR=build_model_check
ANDROID_DIR=/data/local/tmp/model_check
ANDROID_DATA_DIR=$ANDROID_DIR/data
DUMP_DIR=$WORK_DIR/dump_data_model_check
ADB=adb

DEVICE="ARM"
ARMV82="ON"
TEST_PROTO_PATH=
INPUT_PATH=
OPTION_DUMP_OUTPUT=""
OPTION_CHECK_BATCH=
OPTION_CHECK_OUTPUT=
SET_PRECISION=
REFERENCE_PATH=""
OPTION_REFERENCE_FILE=""

function usage() {
    echo "usage: ./model_check_android.sh  [-32] [-v82] [-c] [-b] [-d] <device-id> [-t] <CPU/GPU> [-m] <tnnproto> [-i] <input_file> [-p] [-o] [-s <AUTO/...>]"
    echo "options:"
    echo "        -32   Build 32 bit."
    echo "        -v82  enable armv8.2."
    echo "        -c    Clean up build folders."
    echo "        -b    build targets only"
    echo "        -d    run with specified device"
    echo "        -t    ARM/OPENCL/HUAWEI_NPU specify the platform to run (default: ARM)"
    echo "        -m    tnnproto"
    echo "        -i    input file (NCHW Float)"
    echo "        -f    the reference output file to compare"
    echo "        -p    Push models to device"
    echo "        -do   specify the dump output path(eg: /data/local/tmp/model_check/output_dump)"
    echo "        -a    check multi batch"
    echo "        -e    only check output(precision: AUTO)"
    echo "        -s    AUTO/NORMAL/HIGH/LOW specify the tnn precision(default: HIGH)"
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

    if [ "$DEVICE" == "HUAWEI_NPU" ]
    then
        echo "NPU Enable"
        STL="c++_shared"
        HUAWEI_NPU="ON"

        #start to cp
        if [ ! -d ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/ ]; then
             mkdir -p ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/
        fi
        mkdir -p ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/armeabi-v7a
        mkdir -p ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/arm64-v8a
        cp $ANDROID_NDK/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a/libc++_shared.so  ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/armeabi-v7a/
        cp $ANDROID_NDK/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so ${WORK_DIR}/../../third_party/huawei_npu/cpp_lib/arm64-v8a/
    else
        HUAWEI_NPU="OFF"
    fi

    mkdir -p $BUILD_DIR
    cd $BUILD_DIR || exit
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
          -DTNN_ARM82_ENABLE:BOOL=$ARMV82 \
          -DTNN_HUAWEI_NPU_ENABLE:BOOL=$HUAWEI_NPU \
          -DTNN_MODEL_CHECK_ENABLE:BOOL="ON" \
          -DBUILD_FOR_ANDROID_COMMAND=true
    make -j4
}

function run_android() {
    build_android
    if [ $? != 0 ];then
        echo "build failed"
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

    $ADB shell "rm $ANDROID_DIR/cpu_*.txt"
    $ADB shell "rm $ANDROID_DIR/device_*.txt"

    if [ "" != "$PUSH_MODEL" ]; then
        if [ -z "$TEST_PROTO_PATH" ]
        then
            TEST_PROTO_PATH=../../model/SqueezeNet/squeezenet_v1.1.tnnproto
        fi

        $ADB shell "rm -r $ANDROID_DATA_DIR"
        $ADB shell "mkdir -p $ANDROID_DATA_DIR"

        if [ -n "$INPUT_PATH" ]
        then
            echo "push input file"
            $ADB push ${INPUT_PATH} ${ANDROID_DATA_DIR}/input.txt
        fi
        # push reference file
        if [ -n "$REFERENCE_PATH" ]
        then
            echo "push reference file"
            $ADB push ${REFERENCE_PATH} ${ANDROID_DATA_DIR}/reference.txt
            OPTION_REFERENCE_FILE=" -f ${ANDROID_DATA_DIR}/reference.txt "
        fi
        TEST_MODEL_PATH=${TEST_PROTO_PATH/proto/model}
        $ADB push ${TEST_PROTO_PATH} ${ANDROID_DATA_DIR}/test.tnnproto
        $ADB push ${TEST_MODEL_PATH} ${ANDROID_DATA_DIR}/test.tnnmodel
    fi

    $ADB shell "echo \"${DEVICE}\" > $ANDROID_DIR/test_log.txt"
    $ADB shell "echo \"model: ${TEST_PROTO_PATH}\" >> $ANDROID_DIR/test_log.txt"

    if [ "$DEVICE" == "HUAWEI_NPU" ]
    then
        echo "Run Huawei Npu"
        $ADB shell "mkdir -p $ANDROID_DIR/lib"
        $ADB push $WORK_DIR/../../third_party/huawei_npu/cpp_lib/$ABI/* $ANDROID_DIR/lib
        $ADB push $WORK_DIR/../../third_party/huawei_npu/hiai_ddk_latest/$ABI/* $ANDROID_DIR/lib

        if [ -n "$INPUT_PATH" ]
        then
            $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR:${ANDROID_DIR}/lib ./model_check -d $DEVICE -p $ANDROID_DATA_DIR/test.tnnproto -m $ANDROID_DATA_DIR/test.tnnmodel -i $ANDROID_DATA_DIR/input.txt $OPTION_REFERENCE_FILE $OPTION_DUMP_OUTPUT $OPTION_CHECK_BATCH $OPTION_CHECK_OUTPUT $SET_PRECISION >> $ANDROID_DIR/test_log.txt"
        else
            $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR:${ANDROID_DIR}/lib ./model_check -d $DEVICE -p $ANDROID_DATA_DIR/test.tnnproto -m $ANDROID_DATA_DIR/test.tnnmodel $OPTION_DUMP_OUTPUT $OPTION_CHECK_BATCH $OPTION_CHECK_OUTPUT $SET_PRECISION >> $ANDROID_DIR/test_log.txt"
        fi
    else
        if [ -n "$INPUT_PATH" ]
        then
            $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./model_check -d $DEVICE -p $ANDROID_DATA_DIR/test.tnnproto -m $ANDROID_DATA_DIR/test.tnnmodel -i $ANDROID_DATA_DIR/input.txt $OPTION_REFERENCE_FILE $OPTION_DUMP_OUTPUT $OPTION_CHECK_BATCH $OPTION_CHECK_OUTPUT $SET_PRECISION >> $ANDROID_DIR/test_log.txt"
        else
            $ADB shell "cd $ANDROID_DIR ; LD_LIBRARY_PATH=$ANDROID_DIR ./model_check -d $DEVICE -p $ANDROID_DATA_DIR/test.tnnproto -m $ANDROID_DATA_DIR/test.tnnmodel $OPTION_DUMP_OUTPUT $OPTION_CHECK_BATCH $OPTION_CHECK_OUTPUT $SET_PRECISION >> $ANDROID_DIR/test_log.txt"
        fi
    fi

    $ADB pull $ANDROID_DIR/test_log.txt $DUMP_DIR
    $ADB shell "ls $ANDROID_DIR/cpu_*.txt" |xargs -n1 -t -I file $ADB pull file $DUMP_DIR
    $ADB shell "ls $ANDROID_DIR/device_*.txt" |xargs -n1 -t -I file $ADB pull file $DUMP_DIR
    cat $DUMP_DIR/test_log.txt
}

while [ "$1" != "" ]; do
    case $1 in
        -32)
            shift
            ABI="armeabi-v7a"
            ;;
        -v82)
            shift
            ARMV82="ON"
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
        -t)
            shift
            DEVICE="$1"
            shift
            ;;
        -m)
            shift
            TEST_PROTO_PATH="$1"
            shift
            ;;
        -i)
            shift
            INPUT_PATH="$1"
            shift
            ;;
        -f)
            shift
            REFERENCE_PATH="$1"
            shift
            ;;
        -do)
            shift
            OPTION_DUMP_OUTPUT=" -do $1"
            shift
            ;;
        -a)
            shift
            OPTION_CHECK_BATCH=-b
            ;;
        -e)
            shift
            OPTION_CHECK_OUTPUT=-e
            ;;
        -s)
            shift
            SET_PRECISION=" -sp $1"
            shift
            ;;
        *)
            usage
            exit 1
    esac
done

run_android
