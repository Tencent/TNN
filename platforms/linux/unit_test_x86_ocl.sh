#!/bin/bash
OPENCL="ON"
CLEAN=""
BUILD_ONLY=""

WORK_DIR=`pwd`
BUILD_DIR=build
FILTER=""
DUMP_DIR=$WORK_DIR/dump_data

function usage() {
    echo "-c\tClean up build folders."
    echo "-b\tBuild only."
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
    mkdir -p build
    cd $BUILD_DIR
    cmake ../../.. \
          -DCMAKE_BUILD_TYPE=Release \
          -DTNN_TEST_ENABLE:BOOL="ON"  \
          -DTNN_UNIT_TEST_ENABLE:BOOL="ON"  \
          -DTNN_OPENCL_ENABLE:BOOL=$OPENCL
    make -j4
}

function run_x86() {
    build_x86
    if [ $? != 0 ]; then
        echo "build falied!"
        exit 0
    fi
    if [ "" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi
    mkdir -p $DUMP_DIR
    ./test/unit_test/unit_test -dt OPENCL --gtest_filter="*${FILTER}*"
}

while [ "$1" != "" ]; do
    case $1 in
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
        *)
            usage
            exit 1
    esac
done

run_x86
