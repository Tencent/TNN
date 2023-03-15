//
// Created by tencent on 2020-04-30.
//
#include <vector>
#include <android/bitmap.h>
#include "objectdetectorssd_jni.h"
#include "object_detector_ssd.h"
#include "helper_jni.h"
#include "kannarotate.h"
#include "yuv420sp_to_rgb_fast_asm.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>
#include "tnn/utils/mat_utils.h"

static std::shared_ptr<TNN_NS::ObjectDetectorSSD> gDetector;
static int gComputeUnitType = 0;
static int target_height = 300;
static int target_width = 300;
static std::vector<int> target_dims = {1, 3, target_height, target_width};
static jclass clsObjectInfo;
static jmethodID midconstructorObjectInfo;
static jfieldID fidx1;
static jfieldID fidy1;
static jfieldID fidx2;
static jfieldID fidy2;
static jfieldID fidscore;
static jfieldID fidlandmarks;
static jfieldID fidcls;

JNIEXPORT JNICALL jint
TNN_OBJECT_DETECTORSSD(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height,
                             jfloat scoreThreshold, jfloat iouThreshold, jint topk,
                             jint computUnitType) {
    // Reset bench description
    setBenchResult("");
    std::vector<int> nchw = {1, 3, height, width};
    gDetector = std::make_shared<TNN_NS::ObjectDetectorSSD>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/mobilenetv2_ssd_tf_fix_box.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/mobilenetv2_ssd_tf_fix_box.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(),
         modelContent.length());
    gComputeUnitType = computUnitType;

    TNN_NS::Status status = TNN_NS::TNN_OK;
    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->library_path = "";
    option->proto_content = protoContent;
    option->model_content = modelContent;

    LOGI("the device type %d device huawei_npu", gComputeUnitType);
    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        status = gDetector->Init(option);
    } else if (gComputeUnitType == 2) {
        //add for huawei_npu store the om file
        LOGI("the device type  %d device huawei_npu", gComputeUnitType);
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        gDetector->setNpuModelPath(modelPathStr + "/");
        gDetector->setCheckNpuSwitch(false);
        status = gDetector->Init(option);
    } else {
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        status = gDetector->Init(option);
    }

    if (status != TNN_NS::TNN_OK) {
        LOGE("detector init failed %d", (int) status);
        return -1;
    }

    if (clsObjectInfo == NULL) {
        clsObjectInfo = static_cast<jclass>(env->NewGlobalRef(
                env->FindClass("com/tencent/tnn/demo/ObjectInfo")));
        midconstructorObjectInfo = env->GetMethodID(clsObjectInfo, "<init>", "()V");
        fidx1 = env->GetFieldID(clsObjectInfo, "x1", "F");
        fidy1 = env->GetFieldID(clsObjectInfo, "y1", "F");
        fidx2 = env->GetFieldID(clsObjectInfo, "x2", "F");
        fidy2 = env->GetFieldID(clsObjectInfo, "y2", "F");
        fidscore = env->GetFieldID(clsObjectInfo, "score", "F");
        fidlandmarks = env->GetFieldID(clsObjectInfo, "landmarks", "[F");
        fidcls = env->GetFieldID(clsObjectInfo, "class_id", "I");
    }

    return 0;
}

JNIEXPORT JNICALL jboolean
TNN_OBJECT_DETECTORSSD(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
    TNN_NS::ObjectDetectorSSD tmpDetector;
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/mobilenetv2_ssd_tf_fix_box.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/mobilenetv2_ssd_tf_fix_box.tnnmodel");
    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
    option->library_path = "";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    tmpDetector.setNpuModelPath(modelPathStr + "/");
    tmpDetector.setCheckNpuSwitch(true);
    TNN_NS::Status ret = tmpDetector.Init(option);
    return ret == TNN_NS::TNN_OK;
}

JNIEXPORT JNICALL jint TNN_OBJECT_DETECTORSSD(deinit)(JNIEnv *env, jobject thiz) {

    gDetector = nullptr;
    return 0;
}

JNIEXPORT JNICALL jobjectArray TNN_OBJECT_DETECTORSSD(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint view_width, jint view_height, jint rotate) {
    jobjectArray objectInfoArray;
    auto asyncRefDetector = gDetector;
    std::vector<TNN_NS::ObjectInfo> objectInfoList;
    // Convert yuv to rgb
    LOGI("detect from stream %d x %d r %d", width, height, rotate);
    unsigned char *yuvData = new unsigned char[height * width * 3 / 2];
    jbyte *yuvDataRef = env->GetByteArrayElements(yuv420sp, 0);
    int ret = kannarotate_yuv420sp((const unsigned char *) yuvDataRef, (int) width, (int) height,
                                   (unsigned char *) yuvData, (int) rotate);
    env->ReleaseByteArrayElements(yuv420sp, yuvDataRef, 0);
    unsigned char *rgbaData = new unsigned char[height * width * 4];
    yuv420sp_to_rgba_fast_asm((const unsigned char *) yuvData, height, width,
                              (unsigned char *) rgbaData);
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;

    TNN_NS::DimsVector input_dims = {1, 4, width, height};
    TNN_NS::DimsVector resize_dims = {1, 4, 300, 300};
    float scale_h = input_dims[2] / 300.0f;
    float scale_w = input_dims[3] / 300.0f;

    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, input_dims, rgbaData);
    auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, resize_dims);

    TNN_NS::ResizeParam param;
    TNN_NS::MatUtils::Resize(*input_mat, *resize_mat, param, NULL);

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(resize_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);

    asyncRefDetector->ProcessSDKOutput(output);
    objectInfoList = dynamic_cast<TNN_NS::ObjectDetectorSSDOutput *>(output.get())->object_list;
    delete[] yuvData;
    delete[] rgbaData;
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int) status);
        return 0;
    }

    LOGI("object info list size %d", objectInfoList.size());
    // TODO: copy object info list
    if (objectInfoList.size() > 0) {
        objectInfoArray = env->NewObjectArray(objectInfoList.size(), clsObjectInfo, NULL);
        for (int i = 0; i < objectInfoList.size(); i++) {
            jobject objObjectInfo = env->NewObject(clsObjectInfo, midconstructorObjectInfo);
            int landmarkNum = objectInfoList[i].key_points.size();
            LOGI("object[%d] %f %f %f %f score %f landmark size %d", i, objectInfoList[i].x1,
                 objectInfoList[i].y1, objectInfoList[i].x2, objectInfoList[i].y2,
                 objectInfoList[i].score, landmarkNum);
            auto object_preview = objectInfoList[i].AdjustToImageSize(width, height);
            auto object_orig = object_preview.AdjustToViewSize(view_height, view_width, 2);
            env->SetFloatField(objObjectInfo, fidx1, object_orig.x1);
            env->SetFloatField(objObjectInfo, fidy1, object_orig.y1);
            env->SetFloatField(objObjectInfo, fidx2, object_orig.x2);
            env->SetFloatField(objObjectInfo, fidy2, object_orig.y2);
            env->SetFloatField(objObjectInfo, fidscore, object_orig.score);
            env->SetIntField(objObjectInfo, fidcls, object_orig.class_id);
            env->SetObjectArrayElement(objectInfoArray, i, objObjectInfo);
            env->DeleteLocalRef(objObjectInfo);
        }
        return objectInfoArray;
    } else {
        return 0;
    }
}

JNIEXPORT JNICALL jobjectArray TNN_OBJECT_DETECTORSSD(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width,jint height) {
    jobjectArray objectInfoArray;
    int ret = -1;
    AndroidBitmapInfo  sourceInfocolor;
    void*              sourcePixelscolor;

    if (AndroidBitmap_getInfo(env, imageSource, &sourceInfocolor) < 0) {
        return 0;
    }

    if (sourceInfocolor.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return 0;
    }

    if ( AndroidBitmap_lockPixels(env, imageSource, &sourcePixelscolor) < 0) {
        return 0;
    }
    TNN_NS::BenchOption bench_option;
    bench_option.forward_count = 20;
    gDetector->SetBenchOption(bench_option);
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector input_dims = {1, 4, height, width};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, input_dims, sourcePixelscolor);

    TNN_NS::DimsVector target_dims = {1, 4, 300, 300};
    auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims);

    TNN_NS::ResizeParam param;
    TNN_NS::MatUtils::Resize(*input_mat, *resize_mat, param, NULL);

    float scale_h = height / 300.0f;
    float scale_w = width / 300.0f;

    auto asyncRefDetector = gDetector;
    std::vector<TNN_NS::ObjectInfo> objectInfoList;

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(resize_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);
    AndroidBitmap_unlockPixels(env, imageSource);

    objectInfoList = dynamic_cast<TNN_NS::ObjectDetectorSSDOutput*>(output.get())->object_list;

    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int)status);
        return 0;
    }
    LOGI("bench result: %s", asyncRefDetector->GetBenchResult().Description().c_str());
    char temp[128] = "";
    std::string device = "arm";
    if (gComputeUnitType == 1) {
        device = "gpu";
    } else if (gComputeUnitType == 2) {
        device = "huawei_npu";
    }
    sprintf(temp, " device: %s \ntime:", device.c_str());
    std::string computeUnitTips(temp);
    std::string resultTips = std::string(computeUnitTips + asyncRefDetector->GetBenchResult().Description());
    setBenchResult(resultTips);
    LOGI("object info list size %d", objectInfoList.size());
    // TODO: copy object info list
    if (objectInfoList.size() > 0) {
        objectInfoArray = env->NewObjectArray(objectInfoList.size(), clsObjectInfo, NULL);
        for (int i = 0; i < objectInfoList.size(); i++) {
            jobject objObjectInfo = env->NewObject(clsObjectInfo, midconstructorObjectInfo);
            int landmarkNum = objectInfoList[i].key_points.size();
            LOGI("object[%d] %f %f %f %f score %f landmark size %d", i, objectInfoList[i].x1, objectInfoList[i].y1, objectInfoList[i].x2, objectInfoList[i].y2, objectInfoList[i].score, landmarkNum);
            env->SetFloatField(objObjectInfo, fidx1, objectInfoList[i].x1 * scale_w);
            env->SetFloatField(objObjectInfo, fidy1, objectInfoList[i].y1 * scale_h);
            env->SetFloatField(objObjectInfo, fidx2, objectInfoList[i].x2 * scale_w);
            env->SetFloatField(objObjectInfo, fidy2, objectInfoList[i].y2 * scale_h);
            env->SetFloatField(objObjectInfo, fidscore, objectInfoList[i].score);
            env->SetIntField(objObjectInfo, fidcls, objectInfoList[i].class_id);
            env->SetObjectArrayElement(objectInfoArray, i, objObjectInfo);
            env->DeleteLocalRef(objObjectInfo);
        }
        return objectInfoArray;
    } else {
        return 0;
    }
}
