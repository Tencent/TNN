#include "blazeface_align_jni.h"
#include "blazeface_detector.h"
#include "youtu_face_align.h"
#include "face_detect_aligner.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>
#include <tnn/utils/mat_utils.h>
#include <kannarotate-android-lib/include/kannarotate.h>
#include <yuv420sp_to_rgb_fast_asm.h>

static std::shared_ptr<TNN_NS::FaceDetectAligner> gAligner;

static jclass clsFaceInfo;
static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu, 2 is huawei_npu
static int target_width = 128;
static int target_height = 128;

static std::string modelPathStr = "";
static jmethodID midconstructorFaceInfo;
static jfieldID fidx1;
static jfieldID fidy1;
static jfieldID fidx2;
static jfieldID fidy2;
static jfieldID fidkeypoints;

void makeBlazefaceAlignDetectOption(std::shared_ptr<TNN_NS::BlazeFaceDetectorOption> &option,
                                    std::string &lib_path, std::string &proto_content,
                                    std::string &model_content) {
    option->library_path = lib_path;
    option->proto_content = proto_content;
    option->model_content = model_content;
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->input_width = target_width;
    option->input_height = target_height;
//    option->min_score_threshold = 0.75;
    option->min_suppression_threshold = 0.3;
    option->anchor_path = modelPathStr + "/blazeface_anchors.txt";
}

std::shared_ptr<TNN_NS::BlazeFaceDetector> CreateBlazeFaceDetector(JNIEnv *env, jobject thiz, jstring modelPath,
                           jint width, jint height, jint topk,
                           jint computUnitType) {
    auto predictor = std::make_shared<TNN_NS::BlazeFaceDetector>();
    std::string proto_content, model_content, lib_path = "";
    modelPathStr = jstring2string(env, modelPath);
    proto_content = fdLoadFile(modelPathStr + "/blazeface.tnnproto");
    model_content = fdLoadFile(modelPathStr + "/blazeface.tnnmodel");
    LOGI("proto content size %d model content size %d", proto_content.length(),
         model_content.length());
    gComputeUnitType = computUnitType;

    TNN_NS::Status status = TNN_NS::TNN_OK;
    auto option = std::make_shared<TNN_NS::BlazeFaceDetectorOption>();
    makeBlazefaceAlignDetectOption(option, lib_path, proto_content, model_content);

    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        status = predictor->Init(option);
    } else if (gComputeUnitType == 2) {
        //add for huawei_npu store the om file
        LOGI("the device type  %d device huawei_npu", gComputeUnitType);
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        predictor->setNpuModelPath(modelPathStr + "/");
        predictor->setCheckNpuSwitch(false);
        status = predictor->Init(option);
    } else {
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        status = predictor->Init(option);
    }

    if (status != TNN_NS::TNN_OK) {
        LOGE("detector init failed %d", (int) status);
        return nullptr;
    }

    if (clsFaceInfo == NULL) {
        clsFaceInfo = static_cast<jclass>(env->NewGlobalRef(
                env->FindClass("com/tencent/tnn/demo/FaceInfo")));
        midconstructorFaceInfo = env->GetMethodID(clsFaceInfo, "<init>", "()V");
        fidx1 = env->GetFieldID(clsFaceInfo, "x1", "F");
        fidy1 = env->GetFieldID(clsFaceInfo, "y1", "F");
        fidx2 = env->GetFieldID(clsFaceInfo, "x2", "F");
        fidy2 = env->GetFieldID(clsFaceInfo, "y2", "F");
        fidkeypoints = env->GetFieldID(clsFaceInfo, "keypoints", "[[F");
    }

    return predictor;
}

std::shared_ptr<TNN_NS::YoutuFaceAlign> CreateBlazeFaceAlign(JNIEnv *env, jobject thiz, jstring modelPath,
                           jint width, jint height, jint topk,
                           jint computUnitType, int phase) {
    auto predictor = std::make_shared<TNN_NS::YoutuFaceAlign>();
    std::string proto_content, model_content, lib_path = "";
    modelPathStr = jstring2string(env, modelPath);
    if(phase == 1) {
        proto_content = fdLoadFile(modelPathStr + "/youtu_face_alignment_phase1.tnnproto");
        model_content = fdLoadFile(modelPathStr + "/youtu_face_alignment_phase1.tnnmodel");
    } else if(phase == 2) {
        proto_content = fdLoadFile(modelPathStr + "/youtu_face_alignment_phase2.tnnproto");
        model_content = fdLoadFile(modelPathStr + "/youtu_face_alignment_phase2.tnnmodel");
    }

    LOGI("proto content size %d model content size %d", proto_content.length(),
         model_content.length());
    gComputeUnitType = computUnitType;

    TNN_NS::Status status = TNN_NS::TNN_OK;
    auto option = std::make_shared<TNN_NS::YoutuFaceAlignOption>();

    option->proto_content = proto_content;
    option->model_content = model_content;

    option->input_width = target_width;
    option->input_height = target_height;
    //face threshold
    option->face_threshold = 0.5;
    option->min_face_size = 20;
    //model phase
    option->phase = phase;
    //net_scale
    option->net_scale = phase == 1? 1.2 : 1.3;
    //mean pts path
    std::string mean_file_path = phase==1?  modelPathStr + "/mean_pts_phase1.txt" :  modelPathStr + "/mean_pts_phase2.txt";
    option->mean_pts_path = std::move(mean_file_path);


    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        status = predictor->Init(option);
    } else if (gComputeUnitType == 2) {
        //add for huawei_npu store the om file
        LOGI("the device type  %d device huawei_npu", gComputeUnitType);
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        predictor->setNpuModelPath(modelPathStr + "/");
        predictor->setCheckNpuSwitch(false);
        status = predictor->Init(option);
    } else {
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        status = predictor->Init(option);
    }

    if (status != TNN_NS::TNN_OK) {
        LOGE("align init failed %d", (int) status);
        return nullptr;
    }
    if (clsFaceInfo == NULL) {
        clsFaceInfo = static_cast<jclass>(env->NewGlobalRef(
                env->FindClass("com/tencent/tnn/demo/FaceInfo")));
        midconstructorFaceInfo = env->GetMethodID(clsFaceInfo, "<init>", "()V");
        fidx1 = env->GetFieldID(clsFaceInfo, "x1", "F");
        fidy1 = env->GetFieldID(clsFaceInfo, "y1", "F");
        fidx2 = env->GetFieldID(clsFaceInfo, "x2", "F");
        fidy2 = env->GetFieldID(clsFaceInfo, "y2", "F");
        fidkeypoints = env->GetFieldID(clsFaceInfo, "keypoints", "[[F");
    }
    return predictor;
}

JNIEXPORT jint JNICALL TNN_BLAZEFACE_ALIGN(init)(JNIEnv *env, jobject thiz, jstring modelPath,
                                                 jint width, jint height, jfloat scoreThreshold,
                                                 jfloat iouThreshold, jint topk,
                                                 jint computUnitType) {

    gAligner = std::make_shared<TNN_NS::FaceDetectAligner>();

    // Reset bench description
    auto face_detector = CreateBlazeFaceDetector(env, thiz, modelPath,
                                                 width, height, topk,
                                                 computUnitType);
    if(face_detector == nullptr) {
        LOGE("create align phase1 failed \n");
        return -1;
    }

    auto predictor_phase1 = CreateBlazeFaceAlign(env, thiz, modelPath,
                                                 width, height, topk,
                                                 computUnitType, 1);
    if(predictor_phase1 == nullptr) {
        LOGE("create align phase1 failed \n");
        return -1;
    }

    auto predictor_phase2 = CreateBlazeFaceAlign(env, thiz, modelPath,
                                                 width, height, topk,
                                                 computUnitType, 2);
    if(predictor_phase1 == nullptr) {
        LOGE("create align phase2 failed \n");
        return -1;
    }

    int ret = gAligner->Init({face_detector, predictor_phase1, predictor_phase2});

    return ret;
}

JNIEXPORT JNICALL jboolean
TNN_BLAZEFACE_ALIGN(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
    TNN_NS::BlazeFaceDetector tmpDetector;
    std::string protoContent, modelContent, lib_path = "";
    modelPathStr = jstring2string(env, modelPath);
    protoContent = fdLoadFile(modelPathStr + "/blazeface.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/blazeface.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(),
         modelContent.length());

    TNN_NS::Status status = TNN_NS::TNN_OK;
    auto option = std::make_shared<TNN_NS::BlazeFaceDetectorOption>();
    makeBlazefaceAlignDetectOption(option, lib_path, protoContent, modelContent);
    option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;

    tmpDetector.setNpuModelPath(modelPathStr + "/");
    tmpDetector.setCheckNpuSwitch(true);
    TNN_NS::Status ret = tmpDetector.Init(option);
    LOGI("THE ret %s\n", ret.description().c_str());
    return ret == TNN_NS::TNN_OK;
}

JNIEXPORT JNICALL jint TNN_BLAZEFACE_ALIGN(deinit)(JNIEnv *env, jobject thiz) {
    gAligner = nullptr;
    return 0;
}

JNIEXPORT JNICALL jobjectArray
TNN_BLAZEFACE_ALIGN(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width,
                                         jint height, jint view_width, jint view_height, jint rotate) {
    jobjectArray faceInfoArray;
    auto asyncRefDetector = gAligner ;
    // Convert yuv to rgb
    LOGI("detect from stream %d x %d r %d", width, height, rotate);
    unsigned char *yuvData = new unsigned char[height * width * 3 / 2];
    jbyte *yuvDataRef = env->GetByteArrayElements(yuv420sp, 0);
    int ret = kannarotate_yuv420sp((const unsigned char*)yuvDataRef, (int)width, (int)height, (unsigned char*)yuvData, (int)rotate);
    env->ReleaseByteArrayElements(yuv420sp, yuvDataRef, 0);
    unsigned char *rgbaData = new unsigned char[height * width * 4];
    yuv420sp_to_rgba_fast_asm((const unsigned char*)yuvData, height, width, (unsigned char*)rgbaData);
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;

    TNN_NS::DimsVector input_dims = {1, 4, width, height};
    TNN_NS::DimsVector resize_dims = {1, 4, target_height, target_width};

    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, input_dims, rgbaData);
    auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, resize_dims);

    TNN_NS::ResizeParam param;
    TNN_NS::MatUtils::Resize(*input_mat, *resize_mat, param, NULL);

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(resize_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);

    asyncRefDetector->ProcessSDKOutput(output);

    delete[] yuvData;
    delete[] rgbaData;
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to align %d", (int) status);
        return 0;
    }

    if (output && dynamic_cast<TNN_NS::YoutuFaceAlignOutput *> (output.get())) {
        TNN_NS::YoutuFaceAlignInfo face = dynamic_cast<TNN_NS::YoutuFaceAlignOutput *>(output.get())->face;


        faceInfoArray = env->NewObjectArray(1, clsFaceInfo, NULL);
        jobject objFaceInfo = env->NewObject(clsFaceInfo, midconstructorFaceInfo);
        int keypointsNum = face.key_points.size();

        auto face_preview = face.AdjustToImageSize(width, height);
        auto face_orig = face_preview.AdjustToViewSize(view_height, view_width, 2);
        //from here start to create point
        jclass cls1dArr = env->FindClass("[F");
        // Create the returnable jobjectArray with an initial value
        jobjectArray outer = env->NewObjectArray(keypointsNum, cls1dArr, NULL);
        for (int j = 0; j < keypointsNum; j++) {
            jfloatArray inner = env->NewFloatArray(2);
            float temp[] = {face_orig.key_points[j].first, face_orig.key_points[j].second};
            env->SetFloatArrayRegion(inner, 0, 2, temp);
            env->SetObjectArrayElement(outer, j, inner);
            env->DeleteLocalRef(inner);
        }
        env->SetObjectField(objFaceInfo, fidkeypoints, outer);
        env->SetObjectArrayElement(faceInfoArray, 0, objFaceInfo);
        env->DeleteLocalRef(objFaceInfo);
        return faceInfoArray;
    }
    return 0;
}
