##!/usr/bin/env bash

export CMAKE=cmake
export CPP_COMPILER=g++
export C_COMPILER=gcc
export PYTHON=`which python3`

CLEAN=""
BUILD_DIR=build

#set -xe

function usage() {
    echo "usage: ./build.sh [-c]"
    echo "options:"
    echo "        -c    Clean up build folders."
}

function clean_build() {
    echo $1 | grep "${BUILD_DIR}\b" > /dev/null
    if [[ "$?" != "0" ]]; then
        die "Warnning: $1 seems not to be a BUILD folder."
    fi
    rm -rf $1
    mkdir -p $1
}

function build() {

	$PYTHON script/detect_dependency.py
	if [ "-c" == "${CLEAN}" ]; then
        clean_build ${BUILD_DIR}
	fi
	mkdir -p ${BUILD_DIR}
	cd ${BUILD_DIR}

	${CMAKE} .. -DCMAKE_CXX_COMPILER=$CPP_COMPILER \
			    -DCMAKE_C_COMPILER=$C_COMPILER \
				-DPYTHON_EXECUTABLE=$PYTHON \

	make -j4
	cp  *.so ../
	cd ../
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

build
