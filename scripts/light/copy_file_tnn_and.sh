#!/bin/bash

SHELL_BASE_DIR=$(dirname "$0")  # .
cd $SHELL_BASE_DIR

if [ ! -e "$1" ]; then
    echo "usage: ./copy_file_tnn_and.sh path_to_lightcv_root"
    exit 1
fi

LIGHT_CV_ROOT_DIR=$1

# 拷贝arm64-v8a
target_path=$LIGHT_CV_ROOT_DIR"/GYAIThirdPartyLib/TNN/libs/android/arm64-v8a/"
if [ -d $target_path ]; then
    cp -rf "release_so/arm64-v8a/" $target_path
    cp -rf "release_a/arm64-v8a/" $target_path
else
    echo "arm64-v8a folder not found!: "$target_path
fi

# 拷贝armeabi-v7a
target_path=$LIGHT_CV_ROOT_DIR"/GYAIThirdPartyLib/TNN/libs/android/armeabi-v7a/"
if [ -d $target_path ]; then
    cp -rf "release_so/armeabi-v7a/" $target_path
    cp -rf "release_a/armeabi-v7a/" $target_path
else
    echo "armeabi-v7a folder not found!: "$target_path
fi
