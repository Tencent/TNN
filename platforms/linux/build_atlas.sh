#!/bin/bash

DEBUG=0
ATLAS="ON"
ARM="ON"
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

function build_atlas() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p build
    cd $BUILD_DIR
    cmake ../../.. \
          -DCMAKE_BUILD_TYPE=Debug \
          -DDEBUG=$DEBUG \
          -DLINUX_TEST_ENABLE=1 \
          -DTNN_CPU_ENABLE:BOOL="ON"  \
          -DTNN_X86_ENABLE:BOOL="OFF"  \
          -DTNN_ARM_ENABLE:BOOL=$ARM \
          -DTNN_ATLAS_ENABLE:BOOL=$ATLAS
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

build_atlas
