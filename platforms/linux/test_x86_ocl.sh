#!/bin/bash
DEBUG=0
OPENCL="ON"
CLEAN=""
PROFILING="OFF"
WARM_UP_COUNT=10
ITERATOR_COUNT=100

WORK_DIR=`pwd`
BUILD_DIR=build_ocl
MODEL_DIR=$WORK_DIR/models
DUMP_DIR=$WORK_DIR/dump_data
INPUT_FILE_NAME=rpn_in_0_n1_c3_h320_w320.txt
#INPUT_FILE_NAME=rpn_in_input_array_n1_c3_h256_w192.txt
TEST_PROTO_PATH=
INPUT_PATH=

function usage() {
    echo "usage: ./test_x86_ocl.sh  [-c] [-b] [-f] -m <tnnproto file path> -i <input file path>"
    echo "options:"
    echo "        -c    Clean up build folders."
    echo "        -b    Build only."
    echo "        -f    build profiling targets "
    echo "        -m    tnnproto"
    echo "        -i    input file"
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

function build_x86() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR || exit
    cmake ../../.. \
          -DCMAKE_BUILD_TYPE=Debug \
          -DDEBUG=$DEBUG \
          -DTNN_TEST_ENABLE:BOOL="ON"  \
          -DTNN_PROFILER_ENABLE:BOOL=${PROFILING} \
          -DTNN_X86_ENABLE:BOOL="OFF"  \
          -DTNN_OPENCL_ENABLE:BOOL=$OPENCL
    make -j4
}

function run_x86() {
    build_x86
    if [ "" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi

    if [ "ON" == $PROFILING ]; then
        WARM_UP_COUNT=0
        ITERATOR_COUNT=1
    fi

    mkdir -p $DUMP_DIR
    if [ -z "$TEST_PROTO_PATH" ]
    then
        TEST_PROTO_PATH=$MODEL_DIR/test.tnnproto
    fi
    if [ -n "$INPUT_PATH" ]
    then
        INPUT_PATH=$MODEL_DIR/$INPUT_FILE_NAME
    fi
    ./test/TNNTest -dt=OPENCL -mp=$TEST_PROTO_PATH -ip=$INPUT_PATH -op=$DUMP_DIR/dump_data.txt -wc=$WARM_UP_COUNT -ic=$ITERATOR_COUNT
}

while [ "$1" != "" ]; do
    case $1 in
        -c)
            shift
            CLEAN="-c"
            ;;
        -f)
            shift
            PROFILING="ON"
            ;;
        -b)
            shift
            BUILD_ONLY="-b"
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
        *)
            usage
            exit 1
    esac
done

run_x86
