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
    echo $1 | grep -e "${BUILD_DIR}\b" -e "${BIN_DIR}\b" >/dev/null
    if [[ "$?" != "0" ]]; then
        echo "Warnning: $1 seems not to be a BUILD folder."
    fi
    rm -rf $1
    mkdir -p $1
}

function build_model_check_and_tnn_converter_and_onnx2tnn() {

	clean_build ${BIN_DIR}

    if [ "-c" == "${CLEAN}" ]; then
        clean_build ${BUILD_DIR}
    fi
    pwd
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}

    cmake ../../.. \
        -DCMAKE_BUILD_TYPE=DEBUG \
        -DDEBUG:BOOL="ON" \
        -DTNN_CPU_ENABLE:BOOL="ON" \
        -DTNN_MODEL_CHECK_ENABLE:BOOL="ON" \
        -DTNN_CONVERTER_ENABLE:BOOL="ON" \
        -DTNN_ONNX2TNN_ENABLE:BOOL="ON" \
        -DTNN_DYNAMIC_RANGE_QUANTIZATION_ENABLE:BOOL="ON" \
        -DTNN_BUILD_SHARED="OFF"

    make -j4

    if [ -f "model_check" ]; then
        cp model_check ../${BIN_DIR}/
        echo "Compiled model_check successfully !"
    else

        echo "Compiled model_check failed !!!"
    fi

    if [ -f "dynamic_range_quantization" ]; then
        cp dynamic_range_quantization ../${BIN_DIR}/
        echo "Compiled dynamic_range_quantization successfully !"
    else
        echo "Compiled dynamic_range_quantization failed !!!"
    fi

    if [ -f "tools/converter/TnnConverter" ]; then
        cp tools/converter/TnnConverter ../${BIN_DIR}/
        echo "Compiled TNNConverter successfully !"
    else
        echo "Compiled TNNConverter failed !!!"
    fi

    #From the date 20210123 on, onnx2tnn is compiled by default with Cmake option TNN_CONVERTER_ENABLE
    #Copy the so file of onnx2tnn from the building directory to tnn/tools/onnx2tnn/onnx-converter/onnx2tnn
    onnx2nn_files=$(ls -U tools/onnx2tnn/onnx-converter/onnx2tnn*.so);
    if [ ${#onnx2nn_files[*]} -ge 1 ]; then
        cp ${onnx2nn_files[0]} ../../onnx2tnn/onnx-converter
        rm ${onnx2nn_files[0]}
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

build_model_check_and_tnn_converter_and_onnx2tnn
