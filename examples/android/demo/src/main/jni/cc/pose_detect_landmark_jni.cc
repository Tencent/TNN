//
// Created by tencent on 2020-12-18.
//
#include "pose_detect_landmark_jni.h"
#include "pose_detect_landmark.h"
#include "blazepose_detector.h"
#include "blazepose_landmark.h"
#include "kannarotate.h"
#include "yuv420sp_to_rgb_fast_asm.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>
#include "tnn/utils/mat_utils.h"

static std::shared_ptr<TNN_NS::PoseDetectLandmark> gDetector;
static std::shared_ptr<TNN_NS::PoseDetectLandmark> gFullBodyDetector;
static std::shared_ptr<TNN_NS::BlazePoseDetector> gBlazePoseDetector;
static std::shared_ptr<TNN_NS::BlazePoseLandmark> gBlazePoseLandmark;
static std::shared_ptr<TNN_NS::BlazePoseLandmark> gBlazePoseFullBodyLandmark;
static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu, 2 is huawei_npu
static jclass clsObjectInfo;
static jmethodID midconstructorObjectInfo;
static jfieldID fidx1;
static jfieldID fidy1;
static jfieldID fidx2;
static jfieldID fidy2;
static jfieldID fidscore;
static jfieldID fidcls;
static jfieldID fidkeypoints;
static jfieldID fidlines;
// Jni functions

JNIEXPORT JNICALL jint TNN_POSE_DETECT_LANDMARK(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint computUnitType)
{
    // Reset bench description
    setBenchResult("");
    gDetector = std::make_shared<TNN_NS::PoseDetectLandmark>();
    gFullBodyDetector = std::make_shared<TNN_NS::PoseDetectLandmark>();
    gBlazePoseDetector = std::make_shared<TNN_NS::BlazePoseDetector>();
    gBlazePoseLandmark = std::make_shared<TNN_NS::BlazePoseLandmark>();
    gBlazePoseFullBodyLandmark = std::make_shared<TNN_NS::BlazePoseLandmark>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/pose_detection.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/pose_detection.tnnmodel");
    LOGI("pose detection proto content size %d model content size %d", protoContent.length(), modelContent.length());
    gComputeUnitType = computUnitType;
    LOGI("device type: %d", gComputeUnitType);

    TNN_NS::Status status = TNN_NS::TNN_OK;
    {
        auto option = std::make_shared<TNN_NS::BlazePoseDetectorOption>();
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        option->library_path  = "";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        option->min_score_threshold = 0.5;
        option->min_suppression_threshold = 0.3;
        if (gComputeUnitType == 1) {
            option->compute_units = TNN_NS::TNNComputeUnitsGPU;
            status = gBlazePoseDetector->Init(option);
        } else if (gComputeUnitType == 2) {
            //add for huawei_npu store the om file
            option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
            gBlazePoseDetector->setNpuModelPath(modelPathStr + "/");
            gBlazePoseDetector->setCheckNpuSwitch(false);
            status = gBlazePoseDetector->Init(option);
        } else {
            option->compute_units = TNN_NS::TNNComputeUnitsCPU;
            status = gBlazePoseDetector->Init(option);
        }

        if (status != TNN_NS::TNN_OK) {
            LOGE("blaze pose detector init failed %d", (int)status);
            return -1;
        }
    }

    protoContent = fdLoadFile(modelPathStr + "/pose_landmark_upper_body.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/pose_landmark_upper_body.tnnmodel");
    LOGI("pose landmark proto content size %d model content size %d", protoContent.length(), modelContent.length());

    {
        auto option = std::make_shared<TNN_NS::BlazePoseLandmarkOption>();
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        option->library_path  = "";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        option->pose_presence_threshold = 0.5;
        option->landmark_visibility_threshold = 0.1;
        if (gComputeUnitType == 1) {
            option->compute_units = TNN_NS::TNNComputeUnitsGPU;
            status = gBlazePoseLandmark->Init(option);
        } else if (gComputeUnitType == 2) {
            //add for huawei_npu store the om file
            option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
            gBlazePoseLandmark->setNpuModelPath(modelPathStr + "/");
            gBlazePoseLandmark->setCheckNpuSwitch(false);
            status = gBlazePoseLandmark->Init(option);
        } else {
            option->compute_units = TNN_NS::TNNComputeUnitsCPU;
            status = gBlazePoseLandmark->Init(option);
        }

        if (status != TNN_NS::TNN_OK) {
            LOGE("blaze pose landmark init failed %d", (int)status);
            return -1;
        }
    }

    protoContent = fdLoadFile(modelPathStr + "/pose_landmark_full_body.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/pose_landmark_full_body.tnnmodel");
    LOGI("pose landmark full body proto content size %d model content size %d", protoContent.length(), modelContent.length());

    {
        auto option = std::make_shared<TNN_NS::BlazePoseLandmarkOption>();
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        option->library_path  = "";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        option->pose_presence_threshold = 0.5;
        option->landmark_visibility_threshold = 0.1;
        option->full_body = true;
        if (gComputeUnitType == 1) {
            option->compute_units = TNN_NS::TNNComputeUnitsGPU;
            status = gBlazePoseFullBodyLandmark->Init(option);
        } else if (gComputeUnitType == 2) {
            //add for huawei_npu store the om file
            option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
            gBlazePoseFullBodyLandmark->setNpuModelPath(modelPathStr + "/");
            gBlazePoseFullBodyLandmark->setCheckNpuSwitch(false);
            status = gBlazePoseFullBodyLandmark->Init(option);
        } else {
            option->compute_units = TNN_NS::TNNComputeUnitsCPU;
            status = gBlazePoseFullBodyLandmark->Init(option);
        }

        if (status != TNN_NS::TNN_OK) {
            LOGE("blaze pose landmark init failed %d", (int)status);
            return -1;
        }
    }

    status = gDetector->Init({gBlazePoseDetector, gBlazePoseLandmark});
    if (status != TNN_NS::TNN_OK) {
        LOGE("pose detector init failed %d", (int)status);
        return -1;
    }

    status = gFullBodyDetector->Init({gBlazePoseDetector, gBlazePoseFullBodyLandmark});
    if (status != TNN_NS::TNN_OK) {
        LOGE("pose full body detector init failed %d", (int)status);
        return -1;
    }

    if (clsObjectInfo == NULL) {
        clsObjectInfo = static_cast<jclass>(env->NewGlobalRef(env->FindClass("com/tencent/tnn/demo/ObjectInfo")));
        midconstructorObjectInfo = env->GetMethodID(clsObjectInfo, "<init>", "()V");
        fidx1 = env->GetFieldID(clsObjectInfo, "x1" , "F");
        fidy1 = env->GetFieldID(clsObjectInfo, "y1" , "F");
        fidx2 = env->GetFieldID(clsObjectInfo, "x2" , "F");
        fidy2 = env->GetFieldID(clsObjectInfo, "y2" , "F");
        fidscore = env->GetFieldID(clsObjectInfo, "score" , "F");
        fidcls = env->GetFieldID(clsObjectInfo, "class_id", "I");
        fidkeypoints = env->GetFieldID(clsObjectInfo, "key_points", "[[F");
        fidlines = env->GetFieldID(clsObjectInfo, "lines", "[[I");
    }

    return 0;
}

JNIEXPORT JNICALL jboolean TNN_POSE_DETECT_LANDMARK(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
    TNN_NS::PoseDetectLandmark tmpDetector, tmpFullBodyDetector;
    std::shared_ptr<TNN_NS::BlazePoseDetector> blazePoseDetector = std::make_shared<TNN_NS::BlazePoseDetector>();
    std::shared_ptr<TNN_NS::BlazePoseLandmark> blazePoseLandmark = std::make_shared<TNN_NS::BlazePoseLandmark>();
    std::shared_ptr<TNN_NS::BlazePoseLandmark> blazePoseFullBodyLandmark = std::make_shared<TNN_NS::BlazePoseLandmark>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/pose_detection.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/pose_detection.tnnmodel");
    {
        auto option = std::make_shared<TNN_NS::BlazePoseDetectorOption>();
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        option->library_path = "";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        option->min_score_threshold = 0.5;
        option->min_suppression_threshold = 0.3;
        blazePoseDetector->setNpuModelPath(modelPathStr + "/");
        blazePoseDetector->setCheckNpuSwitch(true);
        TNN_NS::Status ret = blazePoseDetector->Init(option);
        if (ret != TNN_NS::TNN_OK) {
            LOGE("checkNpu failed, ret: %d, msg: %s\n", (int)ret, ret.description().c_str());
            return false;
        }
    }

    protoContent = fdLoadFile(modelPathStr + "/pose_landmark_upper_body.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/pose_landmark_upper_body.tnnmodel");
    {
        auto option = std::make_shared<TNN_NS::BlazePoseLandmarkOption>();
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        option->library_path = "";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        option->pose_presence_threshold = 0.5;
        option->landmark_visibility_threshold = 0.1;
        blazePoseLandmark->setNpuModelPath(modelPathStr + "/");
        blazePoseLandmark->setCheckNpuSwitch(true);
        TNN_NS::Status ret = blazePoseLandmark->Init(option);
        if (ret != TNN_NS::TNN_OK) {
            LOGE("checkNpu failed, ret: %d, msg: %s\n", (int)ret, ret.description().c_str());
            return false;
        }
    }

    protoContent = fdLoadFile(modelPathStr + "/pose_landmark_full_body.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/pose_landmark_full_body.tnnmodel");
    {
        auto option = std::make_shared<TNN_NS::BlazePoseLandmarkOption>();
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        option->library_path = "";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        option->pose_presence_threshold = 0.5;
        option->landmark_visibility_threshold = 0.1;
        option->full_body = true;
        blazePoseFullBodyLandmark->setNpuModelPath(modelPathStr + "/");
        blazePoseFullBodyLandmark->setCheckNpuSwitch(true);
        TNN_NS::Status ret = blazePoseFullBodyLandmark->Init(option);
        if (ret != TNN_NS::TNN_OK) {
            LOGE("checkNpu failed, ret: %d, msg: %s\n", (int)ret, ret.description().c_str());
            return false;
        }
    }

    TNN_NS::Status ret = tmpDetector.Init({blazePoseDetector, blazePoseLandmark});
    if (ret != TNN_NS::TNN_OK) {
        LOGE("checkNpu failed, ret: %d, msg: %s\n", (int)ret, ret.description().c_str());
        return false;
    }

    ret = tmpFullBodyDetector.Init({blazePoseDetector, blazePoseFullBodyLandmark});
    if (ret != TNN_NS::TNN_OK) {
        LOGE("checkNpu failed, ret: %d, msg: %s\n", (int)ret, ret.description().c_str());
        return false;
    }
    return true;
}

JNIEXPORT JNICALL jint TNN_POSE_DETECT_LANDMARK(deinit)(JNIEnv *env, jobject thiz)
{
    gDetector = nullptr;
    gFullBodyDetector = nullptr;
    gBlazePoseDetector = nullptr;
    gBlazePoseLandmark = nullptr;
    gBlazePoseFullBodyLandmark = nullptr;
    return 0;
}

JNIEXPORT JNICALL jobjectArray TNN_POSE_DETECT_LANDMARK(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint view_width, jint view_height, jint rotate, jint detector_type)
{
    jobjectArray objectInfoArray;
    std::shared_ptr<TNN_NS::PoseDetectLandmark> asyncRefDetector;
    if (detector_type == 0) {
        asyncRefDetector = gDetector;
    } else {
        asyncRefDetector = gFullBodyDetector;
    }
    std::vector<TNN_NS::BlazePoseInfo> objectInfoList;
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

    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, input_dims, rgbaData);

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(input_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);

    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int)status);
        return 0;
    }

    if (!output) {
        LOGD("Get empty output\n");
        return 0;
    } else {
        TNN_NS::BlazePoseLandmarkOutput* ptr = dynamic_cast<TNN_NS::BlazePoseLandmarkOutput *>(output.get());
        if (!ptr) {
            LOGD("BlazePose Landmark output empty\n");
            return 0;
        }
    }

    objectInfoList = dynamic_cast<TNN_NS::BlazePoseLandmarkOutput *>(output.get())->body_list;
    delete [] yuvData;
    delete [] rgbaData;
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int)status);
        return 0;
    }

    LOGI("object info list size %d", objectInfoList.size());
    // TODO: copy object info list
    if (objectInfoList.size() > 0) {
        objectInfoArray = env->NewObjectArray(objectInfoList.size(), clsObjectInfo, NULL);
        for (int i = 0; i < objectInfoList.size(); i++) {
            jobject objObjectInfo = env->NewObject(clsObjectInfo, midconstructorObjectInfo);
            auto& objectInfo = objectInfoList[i];
            int keypointsNum = objectInfo.key_points_3d.size();
            int linesNum = objectInfo.lines.size();
            LOGI("object %f %f %f %f score %f key points size %d, label_id: %d, line num: %d",
                    objectInfo.x1, objectInfo.y1, objectInfo.x2,
                    objectInfo.y2, objectInfo.score, keypointsNum, objectInfo.class_id, linesNum);
            auto object_orig = objectInfo.AdjustToViewSize(view_height, view_width, 2);
            //from here start to create point
            jclass cls1dArr = env->FindClass("[F");
            // Create the returnable jobjectArray with an initial value
            jobjectArray outer = env->NewObjectArray(keypointsNum, cls1dArr, NULL);
            for (int j = 0; j < keypointsNum; j++) {
                jfloatArray inner = env->NewFloatArray(2);
                float temp[] = {std::get<0>(object_orig.key_points_3d[j]), std::get<1>(object_orig.key_points_3d[j])};
                env->SetFloatArrayRegion(inner, 0, 2, temp);
                env->SetObjectArrayElement(outer, j, inner);
                env->DeleteLocalRef(inner);
            }

            //from here start to create line
            jclass line1dArr = env->FindClass("[I");
            // Create the returnable jobjectArray with an initial value
            jobjectArray lineOuter = env->NewObjectArray(linesNum, line1dArr, NULL);
            for (int j = 0; j < linesNum; j++) {
                jintArray lineInner = env->NewIntArray(2);
                int temp[] = {object_orig.lines[j].first, object_orig.lines[j].second};
                env->SetIntArrayRegion(lineInner, 0, 2, temp);
                env->SetObjectArrayElement(lineOuter, j, lineInner);
                env->DeleteLocalRef(lineInner);
            }
            env->SetFloatField(objObjectInfo, fidx1, object_orig.x1);
            env->SetFloatField(objObjectInfo, fidy1, object_orig.y1);
            env->SetFloatField(objObjectInfo, fidx2, object_orig.x2);
            env->SetFloatField(objObjectInfo, fidy2, object_orig.y2);
            env->SetFloatField(objObjectInfo, fidscore, object_orig.score);
            env->SetIntField(objObjectInfo, fidcls, object_orig.class_id);
            env->SetObjectField(objObjectInfo, fidkeypoints, outer);
            env->SetObjectField(objObjectInfo, fidlines, lineOuter);
            env->SetObjectArrayElement(objectInfoArray, i, objObjectInfo);
            env->DeleteLocalRef(objObjectInfo);
        }
        return objectInfoArray;
    }

    return 0;
}
