@echo off
SETLOCAL EnableDelayedExpansion

@REM download face-detector tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.tnnmodel" ^
    "face_detector"

@REM download face-detector ncnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.bin" ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/face-detector/version-slim-320_simplified.param" ^
    "face_detector"

@REM download mobilenetv2 tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/mobilenet_v2/mobilenet_v2.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/mobilenet_v2/mobilenet_v2.tnnmodel" ^
    "mobilenet_v2"

@REM download shufflenet_v2 tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/shufflenet_v2/shufflenet_v2.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/shufflenet_v2/shufflenet_v2.tnnmodel" ^
    "shufflenet_v2"

@REM download blazeface tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazeface/blazeface.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/blazeface/blazeface.tnnmodel" ^
    "blazeface"

@REM download blazeface anchor file
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazeface/blazeface_anchors.txt" ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazeface/blazeface_anchors.txt" ^
    "blazeface"

@REM download mobilenet_v2-ssd tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/mobilenet_v2-ssd/mobilenetv2_ssd_tf_fix_box.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/mobilenet_v2-ssd/mobilenetv2_ssd_tf_fix_box.tnnmodel" ^
    "mobilenet_v2-ssd"

@REM download yolov5 tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/yolov5/yolov5s-permute.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/yolov5/yolov5s.tnnmodel" ^
    "yolov5"

@REM download facemesh tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/face_mesh/face_mesh.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/face_mesh/face_mesh.tnnmodel" ^
    "face_mesh"

@REM download YouTu face alignment phase1 tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase1.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase1.tnnmodel" ^
    "youtu_face_alignment"

@REM download YouTu face alignment phase2 tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase2.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase2.tnnmodel" ^
    "youtu_face_alignment"

@REM download YouTu face alignment pts file
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_mean_pts_phase1.txt" ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_mean_pts_phase2.txt" ^
    "youtu_face_alignment"

@REM download hair segmentation tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/hair_segmentation/segmentation.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/hair_segmentation/segmentation.tnnmodel" ^
    "hair_segmentation"

@REM download skeleton big tnn model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/skeleton/skeleton_big.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/skeleton/skeleton.tnnmodel" ^
    "skeleton"

@REM download skeleton medium and small tnn models
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/skeleton/skeleton_middle.tnnproto" ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/skeleton/skeleton_small.tnnproto" ^
    "skeleton"

@REM download blazepose detection model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazepose/pose_detection.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/blazepose/pose_detection.tnnmodel" ^
    "blazepose"

@REM download blazepose upper-body landmark model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazepose/pose_landmark_upper_body.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/blazepose/pose_landmark_upper_body.tnnmodel" ^
    "blazepose"

@REM download blazepose full-body landmark model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazepose/pose_landmark_full_body.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/blazepose/pose_landmark_full_body.tnnmodel" ^
    "blazepose"

@REM download reading comprehension model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/bertsquad10/bertsquad10_clean.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/bertsquad10/bertsquad10_clean.tnnmodel" ^
    "bertsquad10"

call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/bertsquad10/vocab.txt" ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/bertsquad10/vocab.txt" ^
    "bertsquad10"

@REM download monodepth pydnet model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/monodepth_pydnet/monodepth_pydnet.tnnproto" ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/monodepth_pydnet/monodepth_pydnet.tnnproto" ^
    "monodepth_pydnet"

@REM download ocr model
call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/chinese-ocr/angle_net.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/chinese-ocr/angle_net.tnnmodel" ^
    "chinese-ocr"

call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/chinese-ocr/crnn_lite_lstm.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/chinese-ocr/crnn_lite_lstm.tnnmodel" ^
    "chinese-ocr"

call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/chinese-ocr/dbnet.tnnproto" ^
    "https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/chinese-ocr/dbnet.tnnmodel" ^
    "chinese-ocr"

call:download_model ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/chinese-ocr/keys.txt" ^
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/chinese-ocr/keys.txt" ^
    "chinese-ocr"

goto end

:download_model
set directory=%~3
if not exist %directory% (
    md %directory%
)

for /F %%i in ("%~1") do set proto_name=%%~nxi
set proto_path=%directory%\%proto_name%
if not exist %proto_path% (
    echo downloading %~1
    certutil -urlcache -split -f %~1 %proto_path%
    if %errorlevel% neq 0 (
        echo "download file %~1 failed."
        goto end
    )
)

for /F %%i in ("%~2") do set model_name=%%~nxi
set model_path=%directory%\%model_name%
if not exist %model_path% (
    echo downloading %~2
    certutil -urlcache -split -f %~2 %model_path%
    if %errorlevel% neq 0 (
        echo "download file %~2 failed."
        goto end
    )
)

goto:eof

:end
    echo downloading success!