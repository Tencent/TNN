#!/bin/bash

SHARED_LIB="OFF"
CLEAN=""

BUILD_DIR=build

function usage() {
    echo "-c\tClean up build folders."
}

function clean_build() {
    echo $1 | grep "$BUILD_DIR\b" > /dev/null
    if [[ "$?" != "0" ]]; then
        die "Warnning: $1 seems not to be a BUILD folder."
    fi
    rm -rf $1
    mkdir $1
}

function build_model_convert() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p build
    cd $BUILD_DIR
    cmake ../../.. \
          -DCMAKE_BUILD_TYPE=Release \
          -DTNN_BUILD_SHARED:BOOL=$SHARED_LIB \
          -DTNN_MODEL_CONVERT_ENABLE:BOOL="ON"
    make -j4
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
    esac
done

build_model_convert
