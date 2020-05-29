#!/bin/bash

export PATH=$PATH:$ANDROID_HOME/platform-tools

ABI="arm64-v8a"

WORK_DIR=`pwd`
BENCHMARK_MODEL_DIR=$WORK_DIR/../benchmark-model
BUILD_DIR=build
ANDROID_DIR=/data/local/tmp
ANDROID_DATA_DIR=$ANDROID_DIR/benchmark-model
LOOP_COUNT=16
WARM_UP_COUNT=8
benchmark_opencl_list=("squeezenet_v1.0.onnx.opt.onnx.rapidproto" \
                       "mobilenet_v1.onnx.opt.onnx.rapidproto" \
                       "mobilenet_v2.onnx.opt.onnx.rapidproto" \
                       "resnet50.onnx.opt.onnx.rapidproto" \
                       "inception_v3.onnx.opt.onnx.rapidproto" \
                       "shufflenet_v2_x0.5.onnx.opt.onnx.rapidproto" \
                       "yolov3-tiny.onnx.rapidproto" \
                    )
benchmark_arm_list=("squeezenet_v1.0.onnx.opt.onnx.rapidproto" \
                    "mobilenet_v1.onnx.opt.onnx.rapidproto" \
                    "mobilenet_v2.onnx.opt.onnx.rapidproto" \
                    "resnet50.onnx.opt.onnx.rapidproto" \
                    "inception_v3.onnx.opt.onnx.rapidproto" \
                    "quant_squeezenet_v1.0.rapidproto" \
                    "quant_mobilenet_v1.rapidproto" \
                    "quant_mobilenet_v2.rapidproto" \
                    "quant_resnet50.rapidproto" \
                    "quant_inception_v3.rapidproto" \
                    "shufflenet_v2_x0.5.onnx.opt.onnx.rapidproto" \
                    "yolov3-tiny.onnx.rapidproto" \
                )

function clean_build() {
    echo $1 | grep "$BUILD_DIR\b" > /dev/null
    if [[ "$?" != "0" ]]; then
        die "Warnning: $1 seems not to be a BUILD folder."
    fi
    rm -rf $1
    mkdir $1
}

function build_android_bench() {
    if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
    fi
    mkdir -p build
    cd $BUILD_DIR
    cmake ../../.. \
          -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DANDROID_ABI="${ABI}" \
          -DANDROID_STL=c++_static \
          -DANDROID_NATIVE_API_LEVEL=android-14  \
          -DANDROID_TOOLCHAIN=clang \
          -DTNN_ARM_ENABLE:BOOL=ON \
          -DTNN_OPENCL_ENABLE:BOOL=ON \
          -DTNN_OPENMP_ENABLE:BOOL=ON \
          -DTNN_TEST_ENABLE:BOOL=ON \
          -DTNN_BENCHMARK_MODE:BOOL=ON \
          -DBUILD_FOR_ANDROID_COMMAND=true \
          -DNATIVE_LIBRARY_OUTPUT=.
    make -j4
}

function bench_android() {
    build_android_bench
    adb shell "mkdir -p $ANDROID_DIR"
    find . -name "*.so" | while read solib; do
        adb push $solib  $ANDROID_DIR
    done
    adb push test/TNNTest $ANDROID_DIR/TNNTest
    adb shell chmod 0777 $ANDROID_DIR/TNNTest

    adb shell "mkdir -p $ANDROID_DIR/benchmark-model"
    adb push ${BENCHMARK_MODEL_DIR} $ANDROID_DIR

    adb shell "getprop ro.product.model > ${ANDROID_DIR}/benchmark_models_result.txt"
    device=ARM
    adb shell "echo '\nbenchmark device: ${device} \n' >> ${ANDROID_DIR}/benchmark_models_result.txt"
    for benchmark_model in ${benchmark_arm_list[*]}
    do
        adb shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=. ./TNNTest -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mp ${ANDROID_DATA_DIR}/${benchmark_model}  >> benchmark_models_result.txt"
    done

    device=OPENCL
    adb shell "echo '\nbenchmark device: ${device} \n' >> ${ANDROID_DIR}/benchmark_models_result.txt"
    for benchmark_model in ${benchmark_opencl_list[*]}
    do
        adb shell "cd ${ANDROID_DIR}; LD_LIBRARY_PATH=. ./TNNTest -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mp ${ANDROID_DATA_DIR}/${benchmark_model}  >> benchmark_models_result.txt"
    done
    adb pull $ANDROID_DIR/benchmark_models_result.txt ../benchmark_models_result.txt
}

while [ "$1" != "" ]; do
    case $1 in
        -32)
            shift
            ABI="armeabi-v7a with NEON"
            ;;
        -c)
            shift
            CLEAN="-c"
            ;;
        *)
            exit 1
    esac
done

bench_android
