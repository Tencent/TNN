#!/bin/bash

BENCHMARK_DIR=benchmark_app/benchmark/
THREAD_NUM=1
ADB=adb
MODEL_TYPE=TNN
LOOP_COUNT=16
WARM_UP_COUNT=8
SLEEP_INTERVAL=3

benchmark_model_list=(
#test.tnnproto \
mobilenet_v2.tnnproto
)

function usage() {
    echo "usage: ./benchmark_models_app.sh [-th] <thread-num> [-n] [-d] <device-id> [-t] <CPU/GPU>"
    echo "options:"
    echo "        -th   Thread num to run"
    echo "        -n    use ncnn model"
    echo "        -d    run with specified device"
    echo "        -t    CPU/GPU/HUAWEI_NPU specify the platform to run"
}

function exit_with_msg() {
    echo $1
    exit 1
}

function set_up_environment() {
    echo -e "Java environment:\nJAVA_HOME: ${JAVA_HOME}, JRE_HOME: ${JRE_HOME}"
    echo -e "Android environment:\nANDROID_HOME: ${ANDROID_HOME}"
    cd ${BENCHMARK_DIR}
    ./gradlew installDebug
}

function bench_android() {
    set_up_environment

    if [ $? != 0 ];then
        exit_with_msg "set up environment failed"
    fi

    if [ ${#benchmark_model_list[*]} == 0 ]; then
        benchmark_model_list=`ls *.tnnproto`
    fi

    if [ "$DEVICE_TYPE" != "GPU" ] && [ "$DEVICE_TYPE" != "CPU" ]; then
        DEVICE_TYPE=""
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "CPU" ]; then
        device=ARM
        echo -e "\nbenchmark device: ${device}\n"

        for benchmark_model in ${benchmark_model_list[*]}
        do
            TEST_ARGS="-th ${THREAD_NUM} -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mt ${MODEL_TYPE} -mp ${benchmark_model}"
            $ADB shell am start -S \
                -n com.tencent.tnn.benchmark/.MainActivity \
                --es args \'${TEST_ARGS}\' 1 > /dev/null

            sleep $SLEEP_INTERVAL
            $ADB logcat -d | grep "TNN Benchmark time cost" | tail -n 1
        done
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "GPU" ]; then
        device=OPENCL
        echo -e "\nbenchmark device: ${device}\n"

        for benchmark_model in ${benchmark_model_list[*]}
        do
            TEST_ARGS="-th ${THREAD_NUM} -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mt ${MODEL_TYPE} -mp ${benchmark_model}"
            $ADB shell am start -S \
                -n com.tencent.tnn.benchmark/.MainActivity \
                --es args \'${TEST_ARGS}\' 1 > /dev/null 

            sleep $SLEEP_INTERVAL
            $ADB logcat -d | grep "TNN Benchmark time cost" | tail -n 1
        done
    fi
}



while [ "$1" != "" ]; do
    case $1 in
        -d)
            shift
            ADB="adb -s $1"
            shift
            ;;
        -t)
            shift
            DEVICE_TYPE="$1"
            shift
            ;;
        -n)
            shift
            MODEL_TYPE=NCNN
            ;;
        -th)
            shift
            THREAD_NUM=$1
            shift
            ;;
        *)
            usage
            exit 1
    esac
done

bench_android
