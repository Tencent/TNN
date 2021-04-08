#!/bin/bash

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

BUILD_DIR=${TNN_ROOT_PATH}/scripts/build
mkdir -p ${BUILD_DIR}

TNN_INSTALL_DIR=${TNN_ROOT_PATH}/scripts/linux_release
mkdir -p ${TNN_INSTALL_DIR}

pack_tnn() {
    cd ${BUILD_DIR}
    mkdir -p ${TNN_INSTALL_DIR}/lib

    if [ -d ${TNN_INSTALL_DIR}/include ]
    then 
        rm -rf ${TNN_INSTALL_DIR}/include
    fi 

    cp -RP ${TNN_ROOT_PATH}/include ${TNN_INSTALL_DIR}/
    cp -P libTNN.a* ${TNN_INSTALL_DIR}/lib
    #cp test/TNNTest ${TNN_INSTALL_DIR}/bin
}

# 编译 TNN
echo "Configuring TNN ..."
cd ${BUILD_DIR}
cmake ${TNN_ROOT_PATH} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_BUILD_SHARED=OFF

echo "Building TNN ..."
make -j4

if [ 0 -ne $? ]
then
    exit -1
fi

pack_tnn

echo "Done"
