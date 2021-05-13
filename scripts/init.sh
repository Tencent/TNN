#!/bin/bash

# YouTu Tool generated file: DO NOT EDIT!
# According to "templateVersion" in "youtu.json"!

SCRIPT_VERSION="v2.1.16"

# Set sdk build constants. Must comply with project structure specifications.
# @see: https://git.code.oa.com/yt-sdk/yt-sdk-docs/tree/master/project
function setConstants() {
    PROJECT_NAME="light-tnn"

    # build scripts options, see @usage function
    IN_DOCKER="false"
    BUILD_TYPES=("Release")
    BUILD_TEST="false"
    BUILD_QUICK="false"
    BUILD_INNER="false"
    BUILD_XUEBAO="false"
    # support android compiler options
    SUPPORT_ANDROID_TOOLCHAINS="gcc|clang|"
    SUPPORT_ANDROID_STLS="gnustl_static|gnustl_shared|c++_static|c++_shared|"

    # defulat android compiler arguments
    ANDROID_STL="c++_static"
    ANDROID_ABIS=("armeabi-v7a" "arm64-v8a")
    ANDROID_TOOLCHAIN="clang"
    ANDROID_PLATFORM="android-14"

    # ios compiler arguments
    IOS_PLATFORMS=("OS" "SIMULATOR")

    # mac compiler arguments
    MACOS_PLATFORM="OSX"

    # windows compiler arguments
    WINDOWS_ARCHS=("x86" "x86-64")
    #supported visual studio versions
    SUPPORT_VS_VERSIONS="2015|2017|2019|"
    #supported platform toolsets
    SUPPORT_VS_PLATFORM_TOOLSETS="v140|v141|v142|"
    #supported windows sdk version
    SUPPORT_TARGET_WINDOWS_VERSIONS="8.1|10|"

    VS_VERSION="2019"
    PLATFORM_TOOLSET="v142"
    TARGET_WINDOWS_VERSION="10"

    PROJECT_ROOT_PATH="$(cd "$(dirname "${SCRIPT_PATH}")"; pwd -P)"
    CORE_SRC_PATH="${PROJECT_ROOT_PATH}/src"
    ANDROID_PATH="${PROJECT_ROOT_PATH}/android"
    IOS_PATH="${PROJECT_ROOT_PATH}/ios"
    DOCS_PATH="${PROJECT_ROOT_PATH}/docs"
    MODELS_PATH="${PROJECT_ROOT_PATH}/models"
    OUTPUT_PATH="${PROJECT_ROOT_PATH}/output"
    OUTPUT_ANDROID_PATH="${PROJECT_ROOT_PATH}/output/android"
    OUTPUT_IOS_PATH="${PROJECT_ROOT_PATH}/output/ios"
    OUTPUT_MACOS_PATH="${PROJECT_ROOT_PATH}/output/macos"
    OUTPUT_WINDOWS_PATH="${PROJECT_ROOT_PATH}/output/windows"
    OUTPUT_LINUX_PATH="${PROJECT_ROOT_PATH}/output/linux"
    OUTPUT_DV300_PATH="${PROJECT_ROOT_PATH}/output/dv300"
    OUTPUT_RK1808_PATH="${PROJECT_ROOT_PATH}/output/rk1808"

    LOG_INFO="info"
    LOG_ERROR="error"
    LOG_WARN="warn"

    # 只有项目目录下才能获取 git tag
    cd ${PROJECT_ROOT_PATH}
    GIT_TAG="$(git describe --tags --always)"
    GIT_COMMIT_HASH="$(eval "git log -1 --pretty=format:'%h'")"
}

# Use colors, but only if connected to a terminal, and that terminal supports them.
# copy from: https://github.com/robbyrussell/oh-my-zsh/blob/master/tools/install.sh
function setColor() {
    if which tput >/dev/null 2>&1; then
        ncolors=$(tput colors)
    fi

    if [ -t 1 ] && [ -n "$ncolors" ] && [ "$ncolors" -ge 8 ]; then
        RED="$(tput setaf 1)"
        GREEN="$(tput setaf 2)"
        YELLOW="$(tput setaf 3)"
        BLUE="$(tput setaf 4)"
        MAGENTA="$(tput setaf 5)"
        CYAN="$(tput setaf 6)"
        WHIHE="$(tput setaf 7)"
        BOLD="$(tput bold)"
        NORMAL="$(tput sgr0)"
    else
        RED=""
        GREEN=""
        YELLOW=""
        BLUE=""
        MAGENTA=""
        CYAN=""
        WHIHE=""
        BOLD=""
        NORMAL=""
    fi
}

function log() {
    if [ $# != 2 ]; then return; fi

    time=`date '+%Y-%m-%d %H:%M:%S'`
    case $1 in
        $LOG_INFO) printf "${CYAN}[${time}] [INFO] ${2}${NORMAL}" ;;
        $LOG_ERROR) printf "${RED}[${time}] [ERROR] ${2}${NORMAL}" ;;
        $LOG_WARN) printf "${YELLOW}[${time}] [WARN] ${2}${NORMAL}" ;;
        *) printf ${2} ;;
    esac
}

# clean 参数为目标文件夹，若目标文件夹存在，则删除并重新 mkdir
function clean() {
    for p in "$@"
    do
        if [ ! -d ${p} ]; then
            continue
        fi

        # Do not use flag -f.
        rm -r ${p}
        if [ $? -ne 0 ]; then
            log $LOG_ERROR "rm ${p}\n"
            exit 1
        fi
    done
    mkdir -p $@
}

function version() {
    cd $1

    if [ -f "version.txt" ]; then
        rm -r version.txt
    fi

    # get md5 of each file in dir $1
    if type md5sum >/dev/null 2>&1; then
        MD5=$(find "." -type f -exec md5sum "{}" +)
    elif which md5 >/dev/null 2>&1; then
        MD5=$(find "." -type f -exec md5 "{}" +)
    fi

    if [ $1 = $OUTPUT_ANDROID_PATH ]; then
        VERSION_ANDROID_STL="ANDROID_STL: ${ANDROID_STL}\n"
    fi

    SDK_INFO="NAME: ${PROJECT_NAME}
VERSION: ${GIT_TAG}
TIME: $(date '+%Y-%m-%d %H:%M:%S')
BUILD_TYPES: ${BUILD_TYPES}
BUILD_INNER: ${BUILD_INNER}
${VERSION_ANDROID_STL}
MD5:
${MD5}
"
    printf "\n${YELLOW}${SDK_INFO}${NORMAL}\n"
    printf "${SDK_INFO}" > version.txt

    cd -
}

function success() {
    log $LOG_INFO "Build ${1} result: ${GREEN}SUCCESS${NORMAL}
BUILD_TYPES: ${BUILD_TYPES}
BUILD_TEST: ${BUILD_TEST}
BUILD_QUICK: ${BUILD_QUICK}
BUILD_INNER: ${BUILD_INNER}
BUILD_XUEBAO: ${BUILD_XUEBAO}
"
}

# $1: 构建平台，如 Windows/Android/iOS/Linux 等，必须
# $2: 模型文件夹名称，如 face-feature-v7114，可以为空
function _pack() {
    # 必须要 cd 到 OUTPUT_PATH 目录，否则 tar 指定目录会出错
    cd ${OUTPUT_PATH}

    # 注意，这里会对获得的模型目录按照分隔符 "-" 删除所有的前置内容，所以
    # face-feature-v7114 是合格的模型文件夹名称
    # face-feature-v7114-r20190101 将会造成错误
    # 应该改为：face-feature-v7114 或者 face-feature-v7114_r20190101
    local PLATFORM=$1
    if [ "${BUILD_INNER}" == "true" ]; then
        PLATFORM=${PLATFORM}-inner
    fi
    local MODEL_DIRNAME=$2
    local MODEL_VERSION=(${MODEL_DIRNAME//*-/})
    local DIRNAME="${PROJECT_NAME}-${PLATFORM}-${GIT_TAG}"
    if [ -n "${MODEL_VERSION}" ]; then
        DIRNAME="${PROJECT_NAME}-${PLATFORM}-${GIT_TAG}-${MODEL_VERSION}"
    fi
    # SHOULD LIKE YTFaceFeature-Android-v1.0.1-v7114.tar.gz
    # or YTFaceFeature-Android-12cd75d-v7114.tar.gz
    local TARNAME="${DIRNAME}.tar.gz"
    local OUTPUT=""
    case "${PLATFORM}" in
        "Windows"*)
            OUTPUT=${OUTPUT_WINDOWS_PATH}
            ;;
        "msvc"*)
            OUTPUT=${OUTPUT_WINDOWS_PATH}
            ;;            
        "Android"*)
            OUTPUT=${OUTPUT_ANDROID_PATH}
            ;;
        "iOS"*)
            OUTPUT=${OUTPUT_IOS_PATH}
            ;;
        "macOS"*)
            OUTPUT=${OUTPUT_MACOS_PATH}
            ;;
        "Linux"*)
            OUTPUT=${OUTPUT_LINUX_PATH}
            ;;
        "dv300"*)
            OUTPUT=${OUTPUT_DV300_PATH}
            ;;
        "rk1808"*)
            OUTPUT=${OUTPUT_RK1808_PATH}
            ;;
        *)
            log $LOG_ERROR "not support ${PLATFORM}\n"
            exit 1
    esac

    log $LOG_INFO "start tar ${TARNAME} from ${OUTPUT}\n"

    # 拷贝文档
    #rm -rf ${OUTPUT}/docs
    #cp -rv ${DOCS_PATH} ${OUTPUT}/docs

    # 只打包当前版本需要的模型目录
    #if [ -n "${MODEL_VERSION}" ]; then
    #    clean ${OUTPUT}/models
    #    cp -rv ${MODELS_PATH}/${MODEL_DIRNAME} ${OUTPUT}/models
    #fi

    # ios 下将模型目录转换为 bundle
    if [ "${PLATFORM}" == "iOS" ]; then
        mv ${OUTPUT}/models/${MODEL_DIRNAME} ${OUTPUT}/models/${MODEL_DIRNAME}.bundle
    fi

    # 获取 md5
    version "${OUTPUT}"

    rm -rf ${DIRNAME}
    # cp: -a 能够保留其中的软连接，framework 需要软连接
    cp -a ${OUTPUT} ${DIRNAME}
    tar -czf ${TARNAME} ${DIRNAME}
    if [ $? -ne 0 ]; then
        log $LOG_ERROR "tar ${TARNAME} error\n"
        exit 1
    fi

    rm -rf ${DIRNAME}
    # 多次 _pack 中会删掉部分模型，因此这里重新拷贝模型，方便开发调试
    #cp -r ${MODELS_PATH}/* ${OUTPUT}/models/
}

# $1: 构建平台，如 Windows/Android/iOS/Linux 等，必须
function pack() {
    # 模型目录不存在，则直接打包
    if [ ! -d "${MODELS_PATH}" ]; then
        _pack "$1"
        return 0
    fi

    # 获取模型目录下有多少个模型
    MODEL_DIRS=$(cd ${MODELS_PATH}; ls -d *)
    if [ $? -ne 0 ]; then
        _pack "$1"
        return 0
    fi

    for MODEL_DIR in $MODEL_DIRS
    do
        _pack "$1" "${MODEL_DIR}"
    done
}

function init() {
    setColor
    setConstants
}

# 一定要先初始化参数，再读取命令行参数
init

function usage() {
    echo "用法："
    echo "    build-[platform].sh [options]"
    echo "例子："
    echo "    bash ./build-android.sh --android-toolchain gcc --android-stl gnustl_static"
    echo "    bash ./build-ios.sh --debug --quick"
    echo "参数: "
    echo "    -h, --help  输出当前帮助信息"
    echo "    -d, --debug cmake 使用 debug 模式构建，将不会进行鉴权操作"
    echo "    -t, --test  编译 test 可执行文件，请确保 test/test.cpp 文件存在"
    echo "    -q, --quick 不清空 cmake 产生的构建中间物，加速下一次构建"
    echo "    -i, --inner 内部版本，生成静态库（.a）作为中间产物，非必要情况，不要提供客户静态库"
    echo "    -x, --xuebao 使用雪豹编译器编译混淆后的版本，仅支持 android"
    echo "    -do, --docker 在docker环境内编译dv300、rk1808需要增加该flag"
    echo "配置: "
    echo "    -at, --android-toolchain 配置 android 编译器类型，默认为 ${ANDROID_TOOLCHAIN}，可选 ${SUPPORT_ANDROID_TOOLCHAINS}"
    echo "    -as, --android-stl 配置 android STL库类型，默认为 ${ANDROID_STL}，可选 ${SUPPORT_ANDROID_STLS}"
    echo "    -vs, --visual-studio 设置 Visual Studio版本，默认为 ${VS_VERSION}，可选 ${SUPPORT_VS_VERSIONS}"
    echo "    -ts, --platform-toolset 设置 Visual Studio平台工具集，默认为 ${PLATFORM_TOOLSET}，可选 ${SUPPORT_VS_PLATFORM_TOOLSETS}"
    echo "    -tw, --target-windows 设置编译的目标Windows版本，默认为 ${TARGET_WINDOWS_VERSION}，可选 ${SUPPORT_TARGET_WINDOWS_VERSIONS}"
    exit -1
}

# bash 从命令行获取参数的方法
# @see: https://stackoverflow.com/a/33826763/5846160
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help)  usage; shift ;;
        -d|--debug) BUILD_TYPES=("Release" "Debug"); shift ;;
        -t|--test)  BUILD_TEST="true"; shift ;;
        -q|--quick) BUILD_QUICK="true"; shift ;;
        -i|--inner) BUILD_INNER="true"; shift ;;
        -x|--xuebao) BUILD_XUEBAO="true"; shift ;;
        -do|--docker) IN_DOCKER="true"; shift ;;
        -at|--android-toolchain)
            ANDROID_TOOLCHAIN="$2";
            if [[ "$SUPPORT_ANDROID_TOOLCHAINS" != *"$ANDROID_TOOLCHAIN|"* ]]; then
                log $LOG_ERROR "Only support android toolchains: ${SUPPORT_ANDROID_TOOLCHAINS}, not: ${ANDROID_TOOLCHAIN}\n"
                exit 1
            fi
            shift 2 ;;
        -as|--android-stl)
            ANDROID_STL="$2";
            if [[ "$SUPPORT_ANDROID_STLS" != *"$ANDROID_STL|"* ]]; then
                log $LOG_ERROR "Only support android stls: ${SUPPORT_ANDROID_STLS}, not: ${ANDROID_STL}\n"
                exit 1
            fi
            shift 2 ;;
        -vs|--visual-studio)
            VS_VERSION="$2";
            if [[ "$SUPPORT_VS_VERSIONS" != *"$VS_VERSION|"* ]]; then
                log $LOG_ERROR "Supported versions of Visual Studio: ${SUPPORT_VS_VERSIONS}, not: ${VS_VERSION}\n"
                exit 1
            fi
            shift 2 ;;
        -ts|--platform-toolset)
            PLATFORM_TOOLSET="$2";
            if [[ "$SUPPORT_VS_PLATFORM_TOOLSETS" != *"$PLATFORM_TOOLSET|"* ]]; then
                log $LOG_ERROR "Supported platform toolsets: ${SUPPORT_VS_PLATFORM_TOOLSETS}, not: ${PLATFORM_TOOLSET}\n"
                exit 1
            fi
            shift 2 ;;
        -tw|--target-windows)
            TARGET_WINDOWS_VERSION="$2";            
            if [[ "$SUPPORT_TARGET_WINDOWS_VERSIONS" != *"$TARGET_WINDOWS_VERSION|"* ]]; then
                log $LOG_ERROR "Supported target Windows versions: ${SUPPORT_TARGET_WINDOWS_VERSIONS}, not: ${TARGET_WINDOWS_VERSION}\n"
                exit 1
            fi
            shift 2 ;;             
        --) shift; break ;;
        *) break ;;
    esac
done

log $LOG_INFO "Mobile SDK build script version: ${SCRIPT_VERSION}\n"
