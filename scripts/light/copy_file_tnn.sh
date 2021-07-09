#!/bin/bash

SHELL_BASE_DIR=$(dirname "$0")
cd $SHELL_BASE_DIR  # .

if [ ! -e "$1" ]; then
    echo "usage: ./copy_file_tnn.sh path_to_lightcv_root"
    exit 1
fi

LIGHT_CV_ROOT_DIR=$1

# 清理无用文件
useless_file1="build-ios/tnn.framework/CMakeLists.txt"
if [ -f $useless_file1 ]; then
    rm -f $useless_file1
fi

# 拷贝头文件
target_path=$LIGHT_CV_ROOT_DIR"/GYAIThirdPartyLib/TNN/include/tnn/"
if [ -d $target_path ]; then
    cp -rf "build-ios/tnn.framework/Headers/" $target_path
else
    echo "header folder not found!: "$target_path
fi

# 拷贝iOS framework
target_path=$LIGHT_CV_ROOT_DIR"/GYAIThirdPartyLib/TNN/libs/ios/tnn.framework/"
if [ -d $target_path ]; then
    cp -rf "build-ios/tnn.framework/" $target_path
else
    echo "ios framework folder not found!: "$target_path
fi

# 拷贝mac framework
target_path=$LIGHT_CV_ROOT_DIR"/GYAIThirdPartyLib/TNN/libs/mac/tnn.framework/"
if [ -d $target_path ]; then
    cp -rf "build-mac/tnn.framework/" $target_path
else
    echo "mac framework folder not found!: "$target_path
fi

# 拷贝资源bundle, 涉及ios, mac, simulator的metallib
target_path=$LIGHT_CV_ROOT_DIR"/resource/models/tnn.bundle/"
if [ -d $target_path ]; then
    cp -rf "build-ios/tnn.bundle/" $target_path  # ios, simulator
    cp -rf "build-mac/tnn.bundle/" $target_path  # mac
else
    echo "bundle folder not found!: "$target_path
fi
