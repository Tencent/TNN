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
  "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.tnnmodel" \
  "face_detector"

# download face-detector ncnn model
download_model \
  "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.bin" \
  "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.param" \
  "face_detector"

# download mobilenetv2 tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/mobilenet_v2/mobilenet_v2.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/mobilenet_v2/mobilenet_v2.tnnmodel" \
    "mobilenet_v2"

# download shufflenet_v2 tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/shufflenet_v2/shufflenet_v2.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/shufflenet_v2/shufflenet_v2.tnnmodel" \
    "shufflenet_v2"

# download blazeface tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazeface/blazeface.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/blazeface/blazeface.tnnmodel" \
    "blazeface"

# download blazeface anchor file
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazeface/blazeface_anchors.txt" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazeface/blazeface_anchors.txt" \
    "blazeface"

# download mobilenet_v2-ssd tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/mobilenet_v2-ssd/mobilenetv2_ssd_tf_fix_box.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/mobilenet_v2-ssd/mobilenetv2_ssd_tf_fix_box.tnnmodel" \
    "mobilenet_v2-ssd"

# download yolov5 tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/yolov5/yolov5s-permute.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/yolov5/yolov5s.tnnmodel" \
    "yolov5"

# download facemesh tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/face_mesh/face_mesh.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/face_mesh/face_mesh.tnnmodel" \
    "face_mesh"

# download YouTu face alignment phase1 tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase1.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase1.tnnmodel" \
    "youtu_face_alignment"

# download YouTu face alignment phase2 tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase2.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase2.tnnmodel" \
    "youtu_face_alignment"

# download YouTu face alignment pts file
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_mean_pts_phase1.txt" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_mean_pts_phase2.txt" \
    "youtu_face_alignment"

# download hair segmentation tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/hair_segmentation/segmentation.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/hair_segmentation/segmentation.tnnmodel" \
    "hair_segmentation"

# download skeleton big tnn model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/skeleton/skeleton_big.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/skeleton/skeleton.tnnmodel" \
    "skeleton"

# download skeleton medium and small tnn models
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/skeleton/skeleton_middle.tnnproto" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/skeleton/skeleton_small.tnnproto" \
    "skeleton"

# download blazepose detection model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazepose/pose_detection.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/blazepose/pose_detection.tnnmodel" \
    "blazepose"

# download blazepose upper-body landmark model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazepose/pose_landmark_upper_body.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/blazepose/pose_landmark_upper_body.tnnmodel" \
    "blazepose"

# download blazepose full-body landmark model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazepose/pose_landmark_full_body.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/blazepose/pose_landmark_full_body.tnnmodel" \
    "blazepose"

# download reading comprehension model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/bertsquad10/bertsquad10_clean.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/bertsquad10/bertsquad10_clean.tnnmodel" \
    "bertsquad10"
  
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/bertsquad10/vocab.txt" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/bertsquad10/vocab.txt" \
    "bertsquad10"
    

# download ocr model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/chinese-ocr/angle_net.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/chinese-ocr/angle_net.tnnmodel" \
    "chinese-ocr"
  
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/chinese-ocr/crnn_lite_lstm.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/chinese-ocr/crnn_lite_lstm.tnnmodel" \
    "chinese-ocr"
  
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/chinese-ocr/dbnet.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/chinese-ocr/dbnet.tnnmodel" \
    "chinese-ocr"
  
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/chinese-ocr/keys.txt" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/chinese-ocr/keys.txt" \
    "chinese-ocr"

# download nanodet model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/nanodet/nanodet_m.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/nanodet/nanodet_m.tnnmodel" \
    "nanodet"
  
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/nanodet/nanodet_t.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/nanodet/nanodet_t.tnnmodel" \
    "nanodet"
  
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/nanodet/nanodet_e1.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/nanodet/nanodet_e1.tnnmodel" \
    "nanodet"

# download tiny-bert model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/tiny-bert/tiny-bert-squad.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/tiny-bert/tiny-bert-squad.tnnmodel" \
    "tiny-bert"

  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/tiny-bert/tiny-bert-squad-fixed-256.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/tiny-bert/tiny-bert-squad-fixed-256.tnnmodel" \
    "tiny-bert"

  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/tiny-bert/vocab.txt" \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/tiny-bert/vocab.txt" \
    "tiny-bert"

# download monodepth pydnet model
  download_model \
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/monodepth_pydnet/monodepth_pydnet.tnnproto" \
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/monodepth_pydnet/monodepth_pydnet.tnnmodel" \
    "monodepth_pydnet"
