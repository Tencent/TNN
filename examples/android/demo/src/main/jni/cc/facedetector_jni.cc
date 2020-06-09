//
// Created by tencent on 2020-04-29.
//
#include "facedetector_jni.h"
#include "UltraFaceDetector.h"
#include "kannarotate.h"
#include "yuv420sp_to_rgb_fast_asm.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>

static std::shared_ptr<UltraFaceDetector> gDetector;
static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu
static jclass clsFaceInfo;
static jmethodID midconstructorFaceInfo;
static jfieldID fidx1;
static jfieldID fidy1;
static jfieldID fidx2;
static jfieldID fidy2;
static jfieldID fidscore;
static jfieldID fidlandmarks;
// Jni functions

JNIEXPORT JNICALL jint TNN_FACE_DETECTOR(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jfloat scoreThreshold, jfloat iouThreshold, jint topk, jint computUnitType)
{
    // Reset bench description
    setBenchResult("");
    std::vector<int> nchw = {1, 3, height, width};
    gDetector = std::make_shared<UltraFaceDetector>(width, height, 1, 0.7);
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/version-slim-320_simplified.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/version-slim-320_simplified.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());

    TNN_NS::Status status;
    gComputeUnitType = computUnitType;
    if (gComputeUnitType == 0 ) {
        gDetector->Init(protoContent, modelContent, "", TNN_NS::TNNComputeUnitsCPU, nchw);
    } else {
        gDetector->Init(protoContent, modelContent, "", TNN_NS::TNNComputeUnitsGPU, nchw);
    }
    if (status != TNN_NS::TNN_OK) {
        LOGE("detector init failed %d", (int)status);
        return -1;
    }
    TNN_NS::BenchOption bench_option;
    bench_option.forward_count = 1;
    gDetector->SetBenchOption(bench_option);
    if (clsFaceInfo == NULL)
    {
        clsFaceInfo = static_cast<jclass>(env->NewGlobalRef(env->FindClass("com/tencent/tnn/demo/FaceDetector$FaceInfo")));
        midconstructorFaceInfo = env->GetMethodID(clsFaceInfo, "<init>", "()V");
        fidx1 = env->GetFieldID(clsFaceInfo, "x1" , "F");
        fidy1 = env->GetFieldID(clsFaceInfo, "y1" , "F");
        fidx2 = env->GetFieldID(clsFaceInfo, "x2" , "F");
        fidy2 = env->GetFieldID(clsFaceInfo, "y2" , "F");
        fidscore = env->GetFieldID(clsFaceInfo, "score" , "F");
        fidlandmarks = env->GetFieldID(clsFaceInfo, "landmarks" , "[F");
    }

    return 0;
}

JNIEXPORT JNICALL jint TNN_FACE_DETECTOR(deinit)(JNIEnv *env, jobject thiz)
{

    gDetector = nullptr;
    return 0;
}

//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"
JNIEXPORT JNICALL jobjectArray TNN_FACE_DETECTOR(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint rotate)
{
    jobjectArray faceInfoArray;
    auto asyncRefDetector = gDetector;
    std::vector<FaceInfo> faceInfoList;
    // Convert yuv to rgb
    LOGI("detect from stream %d x %d r %d", width, height, rotate);
    unsigned char *yuvData = new unsigned char[height * width * 3 / 2];
    jbyte *yuvDataRef = env->GetByteArrayElements(yuv420sp, 0);
    int ret = kannarotate_yuv420sp((const unsigned char*)yuvDataRef, (int)width, (int)height, (unsigned char*)yuvData, (int)rotate);
    env->ReleaseByteArrayElements(yuv420sp, yuvDataRef, 0);
    unsigned char *rgbaData = new unsigned char[height * width * 4];
    unsigned char *rgbData = new unsigned char[height * width * 3];
    yuv420sp_to_rgba_fast_asm((const unsigned char*)yuvData, height, width, (unsigned char*)rgbaData);
//    stbi_write_jpg(rgba_image_name, height, width, 4, rgbaData, 95);
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector target_dims = {1, 3, height, width};
    auto rgbTNN = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, rgbaData);
    TNN_NS::Status status = asyncRefDetector->Detect(rgbTNN, width, height, faceInfoList);
    delete [] yuvData;
    delete [] rgbaData;
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int)status);
        return 0;
    }

    LOGI("bench result: %s", asyncRefDetector->GetBenchResult().Description().c_str());
    char temp[128] = "";
    sprintf(temp, " device: %s \ntime:", (gComputeUnitType==0)?"arm":"gpu");
    std::string computeUnitTips(temp);
    std::string resultTips = std::string(computeUnitTips + asyncRefDetector->GetBenchResult().Description());
    setBenchResult(resultTips);
    LOGI("face info list size %d", faceInfoList.size());
    // TODO: copy face info list
    if (faceInfoList.size() > 0) {
        faceInfoArray = env->NewObjectArray(faceInfoList.size(), clsFaceInfo, NULL);
        for (int i = 0; i < faceInfoList.size(); i++) {
            jobject objFaceInfo = env->NewObject(clsFaceInfo, midconstructorFaceInfo);
            int landmarkNum = sizeof(faceInfoList[i].landmarks)/sizeof(float);
            LOGI("face[%d] %f %f %f %f score %f landmark size %d", i, faceInfoList[i].x1, faceInfoList[i].y1, faceInfoList[i].x2, faceInfoList[i].y2, faceInfoList[i].score, landmarkNum);
            env->SetFloatField(objFaceInfo, fidx1, faceInfoList[i].x1);
            env->SetFloatField(objFaceInfo, fidy1, faceInfoList[i].y1);
            env->SetFloatField(objFaceInfo, fidx2, faceInfoList[i].x2);
            env->SetFloatField(objFaceInfo, fidy2, faceInfoList[i].y2);
            env->SetFloatField(objFaceInfo, fidscore, faceInfoList[i].score);
//            jfloatArray jarrayLandmarks = env->NewFloatArray(landmarkNum);
//            env->SetFloatArrayRegion(jarrayLandmarks, 0, landmarkNum , faceInfoList[i].landmarks);
//            env->SetObjectField(objFaceInfo, fidlandmarks, jarrayLandmarks);
            env->SetObjectArrayElement(faceInfoArray, i, objFaceInfo);
            env->DeleteLocalRef(objFaceInfo);
        }
        return faceInfoArray;
    } else {
        return 0;
    }

}
JNIEXPORT JNICALL jobjectArray TNN_FACE_DETECTOR(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height)
{
    jobjectArray faceInfoArray;;
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
    TNN_NS::DimsVector target_dims = {1, 3, height, width};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, sourcePixelscolor);
    auto asyncRefDetector = gDetector;
    std::vector<FaceInfo> faceInfoList;
    TNN_NS::Status status = asyncRefDetector->Detect(input_mat, height, width, faceInfoList);
    AndroidBitmap_unlockPixels(env, imageSource);
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int)status);
        return 0;
    }
    LOGI("bench result: %s", asyncRefDetector->GetBenchResult().Description().c_str());
    char temp[128] = "";
    sprintf(temp, " device: %s \ntime:", (gComputeUnitType==0)?"arm":"gpu");
    std::string computeUnitTips(temp);
    std::string resultTips = std::string(computeUnitTips + asyncRefDetector->GetBenchResult().Description());
    setBenchResult(resultTips);
    LOGI("face info list size %d", faceInfoList.size());
    // TODO: copy face info list
    if (faceInfoList.size() > 0) {
        faceInfoArray = env->NewObjectArray(faceInfoList.size(), clsFaceInfo, NULL);
        for (int i = 0; i < faceInfoList.size(); i++) {
            jobject objFaceInfo = env->NewObject(clsFaceInfo, midconstructorFaceInfo);
            int landmarkNum = sizeof(faceInfoList[i].landmarks)/sizeof(float);
            LOGI("face[%d] %f %f %f %f score %f landmark size %d", i, faceInfoList[i].x1, faceInfoList[i].y1, faceInfoList[i].x2, faceInfoList[i].y2, faceInfoList[i].score, landmarkNum);
            env->SetFloatField(objFaceInfo, fidx1, faceInfoList[i].x1);
            env->SetFloatField(objFaceInfo, fidy1, faceInfoList[i].y1);
            env->SetFloatField(objFaceInfo, fidx2, faceInfoList[i].x2);
            env->SetFloatField(objFaceInfo, fidy2, faceInfoList[i].y2);
            env->SetFloatField(objFaceInfo, fidscore, faceInfoList[i].score);
//            jfloatArray jarrayLandmarks = env->NewFloatArray(landmarkNum);
//            env->SetFloatArrayRegion(jarrayLandmarks, 0, landmarkNum , faceInfoList[i].landmarks);
//            env->SetObjectField(objFaceInfo, fidlandmarks, jarrayLandmarks);
            env->SetObjectArrayElement(faceInfoArray, i, objFaceInfo);
            env->DeleteLocalRef(objFaceInfo);
        }
        return faceInfoArray;
    } else {
        return 0;
    }

    return 0;
}
