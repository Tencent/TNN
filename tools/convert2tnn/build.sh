#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
CLEAN=""
BUILD_DIR=build
BIN_DIR=bin

function usage() {
    echo "usage: ./build.sh [-c]"
    echo "options:"
    echo "        -c    Clean up build folders."
}

function clean_build() {
    echo $1 | grep "${BUILD_DIR}\b" >/dev/null
    if [[ "$?" != "0" ]]; then
        die "Warnning: $1 seems not to be a BUILD folder."
    fi
    rm -rf $1
    mkdir -p $1
}

function build_model_check_and_tnn_converter() {

	clean_build ${BIN_DIR}

    if [ "-c" == "${CLEAN}" ]; then
        clean_build ${BUILD_DIR}
    fi
    pwd
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}

    cmake ../../.. \
        -DCMAKE_BUILD_TYPE=DEBUG \
        -DTNN_CPU_ENABLE:BOOL="ON" \
        -DTNN_MODEL_CHECK_ENABLE:BOOL="ON" \
        -DTNN_CONVERTER_ENABLE:BOOL="ON" \
        -DTNN_BUILD_SHARED="OFF" \
        -DDEBUG="ON"

    make -j4

    if [ -f "model_check" ]; then
        cp model_check ../${BIN_DIR}/
        echo "Compiled model_check successfully !"
    else

        echo "Compiled model_check failed !!!"
    fi

    if [ -f "tools/converter/TnnConverter" ]; then
        cp tools/converter/TnnConverter ../${BIN_DIR}/
        echo "Compiled model_check successfully !"
    else
        echo "Compiled TNNConverter failed !!!"
    fi
}

function build_onnx2tnn() {
    cd ${CURRENT_DIR}
    cd ../onnx2tnn/onnx-converter/

    if [ "-c" == "${CLEAN}" ]; then
        ./build.sh -c
    else
        ./build.sh
    fi
}

while [ "$1" != "" ]; do
    case $1 in
    -c)
        shift
        CLEAN="-c"
        ;;
    *)
        usage
        exit 1
        ;;
    esac
done

build_model_check_and_tnn_converter

build_onnx2tnn
