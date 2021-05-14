#!/bin/bash

# YouTu Tool generated file: DO NOT EDIT!
# According to "templateVersion" in "youtu.json"!

SCRIPT_PATH="$(cd "$(dirname "$0")"; pwd -P)"

# `init.sh` will:
# 1. import path variables and module constants
# 2. import utils function
source ${SCRIPT_PATH}/init.sh

# 为了初始化MSBuild，需要设置VS_TOOLS_DIR环境变量以寻找VsDevCmd.bat以及VsMSBuildCmd.bat两个脚本。
# 这两个脚本一般位于<path-to-visual-studio>/Common7/Tools下。

function initMSBuild(){
    if [[ "$VS_TOOLS_DIR" == "" ]]; then
        log $LOG_ERROR "ERROR: environment variable VS_TOOLS_DIR not set yet, exit..."
        exit 1
    fi

    cd "$VS_TOOLS_DIR"
    .\\VsDevCmd.bat
    .\\VsMSBuildCmd.bat
    cd -   
}

# 删除不需要的文件以及整理输出目录
function trimOutputDir(){
    cd ${1}
    mkdir -p ../tmp
    local DESIRED_FILES_EXTENSIONS=("lib" "dll" "pdb" "exp")
    for item in `ls .`
    do
        local EXTENSION=$(echo $item | cut -d '.' -f 2)
        if [[ "${DESIRED_FILES_EXTENSIONS[@]}" =~ "$EXTENSION" ]]; then
            mv $item ../tmp
        fi
    done
    cd ..
    rm -r ${1}
    mv tmp ${1} 
}

function configureCMakeGenerator() {
    case "${1}" in
        "2015")
            GENERATOR="Visual Studio 14 2015";
            if [[ ${2} == "x86-64" ]]; then
                GENERATOR="$GENERATOR Win64"
            fi
            ;;
        "2017")
            GENERATOR="Visual Studio 15 2017";
            if [[ ${2} == "x86-64" ]]; then
                GENERATOR="$GENERATOR Win64"
            fi
            ;;
        "2019")
            GENERATOR="Visual Studio 16 2019";
            if [[ ${2} == "x86-64" ]]; then
                AARCH="-A x64"
            else
                AARCH="-A Win32"
            fi
            ;;
        *)
            log $LOG_ERROR "not support ${1}\n"
            exit 1
    esac    
}

function buildDll(){
    for ARCH in ${WINDOWS_ARCHS[@]}
    do
        for BUILD_TYPE in ${BUILD_TYPES[@]}
        do
            if [[ $BUILD_TYPE == "Release" ]]; then
                BUILD_TYPE="RelWithDebInfo"
            fi
            log $LOG_INFO "start build ${ARCH} ${BUILD_TYPE} libs\n"
            BUILD_PATH=${PROJECT_ROOT_PATH}/build/${ARCH}/${BUILD_TYPE}
            if [ "${BUILD_QUICK}" != "true" ]; then
                clean ${BUILD_PATH}
            else
                [ ! -d "${BUILD_PATH}" ] && mkdir -p ${BUILD_PATH}
            fi
            cd ${BUILD_PATH}

            configureCMakeGenerator "$VS_VERSION" "${ARCH}"
            echo ${PROJECT_ROOT_PATH}
            cmake ${PROJECT_ROOT_PATH} -G "$GENERATOR" $AARCH -T $PLATFORM_TOOLSET \
                -DTNN_CPU_ENABLE=ON \
                -DTNN_TEST_ENABLE=OFF \
                -DINTTYPES_FORMAT=C99 \
                -DTNN_X86_ENABLE=ON \
                -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
                -DWINDOWS_ARCH=${ARCH} \
                -DCMAKE_SYSTEM_NAME=Windows \
                -DCMAKE_SYSTEM_VERSION=${TARGET_WINDOWS_VERSION} \
                -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
                -DBUILD_TEST=${BUILD_TEST} \
                -DBUILD_INNER=${BUILD_INNER} \
                -DOUTPUT_PATH=${OUTPUT_WINDOWS_PATH} \
                -DLIBRARY_OUTPUT_PATH=${OUTPUT_WINDOWS_PATH}/libs/${ARCH} \
                -DEXECUTABLE_OUTPUT_PATH=${OUTPUT_WINDOWS_PATH}/bin/${ARCH}
            if [ $? -ne 0 ]; then
				log $LOG_ERROR "cmake ${ARCH} error\n"
				exit 1
            fi

            MSBuild.exe ALL_BUILD.vcxproj -property:Configuration=${BUILD_TYPE}
            #MSBuild.exe INSTALL.vcxproj -property:Configuration=${BUILD_TYPE}
            #trimOutputDir ${OUTPUT_WINDOWS_PATH}/libs/${ARCH}/${BUILD_TYPE}
            #if [ $? -ne 0 ]; then
            #	log $LOG_ERROR "trimOutput ${ARCH} error\n"
            #	exit 1
            #fi
        done
        cp -r ${PROJECT_ROOT_PATH}/include ${OUTPUT_WINDOWS_PATH}
    done
}

function main(){
    # initMSBuild

    if [ "${BUILD_INNER}" == "true" ]; then
        OUTPUT_WINDOWS_PATH=${OUTPUT_WINDOWS_PATH}-inner
    fi

    log $LOG_INFO "start clean output\n"
    clean "${OUTPUT_WINDOWS_PATH}"

    log $LOG_INFO "start build dll\n"
    buildDll

    log $LOG_INFO "start pack windows\n"
    pack "msvc"
    
    success "Windows"
    printf "${MAGENTA}Visual Studio version: ${VS_VERSION}${NORMAL}\n"
    printf "${MAGENTA}Platform toolset: ${PLATFORM_TOOLSET}${NORMAL}\n"
    printf "${MAGENTA}Target Windows Version: ${TARGET_WINDOWS_VERSION}${NORMAL}\n"
}

main