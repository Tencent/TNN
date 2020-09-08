#!/bin/bash

#URL, local path
download_file() { #URL, path
  if [ -e $2 ]; then return 0; fi

  name=`basename $2`
  echo "downloading $name ..."
  status=`curl $1 -s -w %{http_code} -o $2`
  if (( status == 200 )); then
    return 0
  else
    echo "download $name failed" 1>&2
    return -1
  fi
}

#URL proto, URL model, directory
download_model() {
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
      rm -r ${directory}
    fi
  fi

  model_name=`basename $2`
  model_path_local="${directory}/${model_name}"
  if [ ! -f ${model_path_local} ]; then
    download_file $2 $model_path_local
    succ=$?
    if [ ! $succ -eq 0 ]; then
      rm -r ${directory}
    fi
  fi
}

echo "$(dirname $0)"

# download face-detector tnn model
download_model \
  "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.tnnproto" \
  "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.tnnmodel" \
  "face_detector"

# download face-detector ncnn model
download_model \
  "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.bin" \
  "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.param" \
  "face_detector"

# download mobilenetv2 tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/mobilenet_v2/mobilenet_v2.tnnproto" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/mobilenet_v2/mobilenet_v2.tnnmodel" \
    "mobilenet_v2"

# download shufflenet_v2 tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/shufflenet_v2/shufflenet_v2.tnnproto" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/shufflenet_v2/shufflenet_v2.tnnmodel" \
    "shufflenet_v2"

# download blazeface tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazeface/blazeface.tnnproto" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazeface/blazeface.tnnmodel" \
    "blazeface"

# download mobilenet_v2-ssd tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/mobilenet_v2-ssd/mobilenetv2_ssd.tnnproto" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/mobilenet_v2-ssd/mobilenetv2_ssd.tnnmodel" \
    "mobilenet_v2-ssd"

# download yolov5 tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/yolov5/yolov5s-permute.tnnproto" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/yolov5/yolov5s.tnnmodel" \
    "yolov5"

# download facemesh tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/face_mesh/face_mesh.tnnproto" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/face_mesh/face_mesh.tnnmodel" \
    "face_mesh"
