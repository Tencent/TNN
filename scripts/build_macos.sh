#!/bin/bash

SHARED_LIB="ON"
METAL="ON"

CWD=$(cd `dirname $0`; pwd)
if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=${CWD}/..
fi

function CheckRtnAndPrintMsg()
{
    if [ 0 -ne $? ]
    then
        echo $1' failed.'
        exit -1
    fi
    echo $1' completes.'
}

mkdir build_macos
cd build_macos

cmake ${TNN_ROOT_PATH} \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_METAL_ENABLE:BOOL=$METAL \
    -DTNN_UNIT_TEST_ENABLE=ON \
    -DTNN_COVERAGE=ON \
    -DTNN_BENCHMARK_MODE=ON \
    -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB \
    -DTNN_CONVERTER_ENABLE=ON

make -j4

# check if compiling error occurs, or ci will ignore building errors
CheckRtnAndPrintMsg "building"

echo 'start unit_test'
cd test/unit_test
BUILD_PATH=${CWD}/build_macos
echo "check ${BUILD_PATH}"
if [ ! -f ${BUILD_PATH}/tnn.metallib ]; then
    echo "No metallib found!"
    exit 0
fi
./unit_test --lp ${BUILD_PATH}/tnn.metallib --dt METAL

# check if unit_test error occurs, or ci will ignore building errors
CheckRtnAndPrintMsg "unit_test"
