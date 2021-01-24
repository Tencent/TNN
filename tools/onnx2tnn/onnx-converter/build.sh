##!/usr/bin/env bash

export CMAKE=cmake
export CPP_COMPILER=g++
export C_COMPILER=gcc
export PYTHON=`which python3`

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

function build_onnx2tnn() {

	clean_build ${BIN_DIR}

    if [ "-c" == "${CLEAN}" ]; then
        clean_build ${BUILD_DIR}
    fi
    pwd
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}

    cmake ../../../.. \
        -DCMAKE_BUILD_TYPE=DEBUG \
        -DDEBUG:BOOL="ON" \
        -DTNN_CPU_ENABLE:BOOL="ON" \
        -DTNN_ONNX2TNN_ENABLE:BOOL="ON" \
        -DTNN_BUILD_SHARED="OFF"

    make -j4

    #From the date 20210123 on, onnx2tnn is compiled by default with Cmake option DTNN_CONVERTER_ENABLE
    onnx2nn_files=$(ls -U tools/onnx2tnn/onnx-converter/onnx2tnn*.so);
    if [ ${#onnx2nn_files[*]} -ge 1 ]; then
        cp ${onnx2nn_files[i]} ../../../../../onnx-converter
        rm ${onnx2nn_files[i]}
        echo "Compiled onnx2tnn successfully !"
    else
        echo "Compiled onnx2tnn failed !!!"
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

build_onnx2tnn
