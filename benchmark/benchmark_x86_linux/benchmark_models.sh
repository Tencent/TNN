#!/bin/bash

MODEL_TYPE=TNN
NETWORK_TYPE=OPENVINO
NUM_THREAD=4
BUILD_ONLY="OFF"
DOWNLOAD_MODEL="OFF"

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/../..
fi

WORK_DIR=`pwd`
BENCHMARK_MODEL_DIR=$WORK_DIR/benchmark_model
OUTPUT_LOG_FILE=benchmark_models_result.txt
LOOP_COUNT=20
WARM_UP_COUNT=5

benchmark_model_list=(
#test.tnnproto \
)

#URL, local path
function download_file() { #URL, path
  if [ -e $2 ]; then return 0; fi

  name=`basename $2`
  echo "downloading $name ..."
  # status=`wget $1 -o $2`
  status=`curl $1 -s -w %{http_code} -o $2`
  if (( status == 200 )); then
    return 0
  else
    echo "download $name failed" 1>&2
    return -1
  fi
}

#URL proto, URL model, directory
function download_model() {
  directory="./$3"
  if [ ! -e ${directory} ]; then
    mkdir -p ${directory}
  fi

  proto_name=`basename $1`
  proto_path_local="${directory}/${proto_name}"
  if [ ! -f ${proto_path_local} ]; then
    download_file $1 $proto_path_local
    succ=$?
    if [ ! $succ -eq 0 ]; then
      echo "please download model manually!!!(url:https://github.com/darrenyao87/tnn-models/tree/master/model)"
      rm -r ${directory}
    fi
  fi

  model_name=`basename $2`
  model_path_local="${directory}/${model_name}"
  if [ ! -f ${model_path_local} ]; then
    download_file $2 $model_path_local
    succ=$?
    if [ ! $succ -eq 0 ]; then
      echo "please download model manually!!!(url:https://github.com/darrenyao87/tnn-models/tree/master/model)"
      rm -r ${directory}
    fi
  fi
}

function download_bench_model() {
    download_model \
      "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/resnet50/resnet50.opt.tnnproto" \
      "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/resnet50/resnet50.opt.tnnmodel" \
      benchmark_model

    download_model \
      "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/bert-based/bert-based.tnnproto" \
      "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/bert-based/bert-based.tnnmodel" \
      benchmark_model

    download_model \
      "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/bertsquad10/bertsquad10_clean.tnnproto" \
      "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/bertsquad10/bertsquad10_clean.tnnmodel" \
      benchmark_model
}

function usage() {
    echo "usage: ./benchmark_models.sh  [-th] [-b] [-dl] [-mp] [-native]"
    echo "options:"
    echo "        -th      thread num, defalut 1"
    echo "        -b       build only "
    echo "        -dl      download model from github "
    echo "        -mp      model dir path"
    echo "        -native  bench with native optimization"
}

function exit_with_msg() {
    echo $1
    exit 1
}

function build_x86_linux_bench() {
    cd $TNN_ROOT_PATH/scripts
    ./build_x86_linux.sh
    cp $TNN_ROOT_PATH/scripts/x86_linux_release $TNN_ROOT_PATH/benchmark/benchmark_x86_linux/ -r
}

function bench_x86_linux() {
    if [ "OFF" != "$DOWNLOAD_MODEL" ];then
      download_bench_model
    fi

    build_x86_linux_bench
    if [ $? != 0 ];then
        exit_with_msg "build failed"
    fi

    if [ "OFF" != "$BUILD_ONLY" ]; then
        echo "build done!"
        exit 0
    fi

    if [ ! -d ${BENCHMARK_MODEL_DIR} ]; then
        echo "please set model dir path or exec script with option -dl"
        usage
        exit -1
    fi
    cd ${BENCHMARK_MODEL_DIR}

    if [ ${#benchmark_model_list[*]} == 0 ];then
        benchmark_model_list=`ls *.tnnproto`
    fi

    if [ "$DEVICE_TYPE" = "" ] || [ "$DEVICE_TYPE" = "CPU" ];then
        device=X86
        echo "benchmark device: ${device} " >> $WORK_DIR/$OUTPUT_LOG_FILE

    for benchmark_model in ${benchmark_model_list[*]}
    do
        cd ${WORK_DIR}; LD_LIBRARY_PATH=x86_linux_release/lib ./x86_linux_release/bin/TNNTest -th ${NUM_THREAD} -wc ${WARM_UP_COUNT} -ic ${LOOP_COUNT} -dt ${device} -mt ${MODEL_TYPE} -nt ${NETWORK_TYPE} -mp ${BENCHMARK_MODEL_DIR}/${benchmark_model}  >> $OUTPUT_LOG_FILE
    done
    fi

    echo '' >> $OUTPUT_LOG_FILE
    date  >> $OUTPUT_LOG_FILE

    cat ${WORK_DIR}/$OUTPUT_LOG_FILE
}

while [ "$1" != "" ]; do
    case $1 in
        -native)
            shift
            NETWORK_TYPE=DEFAULT
            ;;
        -th)
            shift
            NUM_THREAD="$1"
            shift
            ;;
        -b)
            shift
            BUILD_ONLY=ON
            ;;
        -dl)
            shift
            DOWNLOAD_MODEL=ON
            ;;
        -mp)
            shift
            BENCHMARK_MODEL_DIR=$(cd $1; pwd)
            shift
            ;;
        *)
            usage
            exit 1
    esac
done

bench_x86_linux
