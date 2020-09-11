// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "face_detector_jni.h"
#include "ultra_face_detector.h"
#include "kannarotate.h"
#include "yuv420sp_to_rgb_fast_asm.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>

static std::shared_ptr<TNN_NS::UltraFaceDetector> gDetector;
static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu, 2 is huawei_npu
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
    gDetector = std::make_shared<TNN_NS::UltraFaceDetector>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/version-slim-320_simplified.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/version-slim-320_simplified.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());
    gComputeUnitType = computUnitType;

    TNN_NS::Status status = TNN_NS::TNN_OK;
    auto option = std::make_shared<TNN_NS::UltraFaceDetectorOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->input_shapes.insert(std::pair<std::string, TNN_NS::DimsVector>("input", nchw));
    option->library_path="";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    option->input_width = width;
    option->input_height= height;
    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        status = gDetector->Init(option);
    } else if (gComputeUnitType == 2) {
        //add for huawei_npu store the om file
        LOGE("the device type  %d device huawei_npu" ,gComputeUnitType);
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        gDetector->setNpuModelPath(modelPathStr + "/");
        gDetector->setCheckNpuSwitch(false);
        status = gDetector->Init(option);
    } else {
	    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    	status = gDetector->Init(option);
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

JNIEXPORT JNICALL jboolean TNN_FACE_DETECTOR(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
    TNN_NS::UltraFaceDetector tmpDetector;
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/version-slim-320_simplified.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/version-slim-320_simplified.tnnmodel");
    auto option = std::make_shared<TNN_NS::UltraFaceDetectorOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
    option->library_path = "";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    option->input_height= 240;
    option->input_width = 320;
    tmpDetector.setNpuModelPath(modelPathStr + "/");
    tmpDetector.setCheckNpuSwitch(true);
    TNN_NS::Status ret = tmpDetector.Init(option);
    return ret == TNN_NS::TNN_OK;
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
    std::vector<TNN_NS::FaceInfo> faceInfoList;
    // Convert yuv to rgb
    LOGI("detect from stream %d x %d r %d", width, height, rotate);
    unsigned char *yuvData = new unsigned char[height * width * 3 / 2];
    jbyte *yuvDataRef = env->GetByteArrayElements(yuv420sp, 0);
    int ret = kannarotate_yuv420sp((const unsigned char*)yuvDataRef, (int)width, (int)height, (unsigned char*)yuvData, (int)rotate);
    env->ReleaseByteArrayElements(yuv420sp, yuvDataRef, 0);
    unsigned char *rgbaData = new unsigned char[height * width * 4];
    yuv420sp_to_rgba_fast_asm((const unsigned char*)yuvData, height, width, (unsigned char*)rgbaData);
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector target_dims = {1, 4, width, height};

    auto rgbTNN = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, rgbaData);
    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(rgbTNN);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);

    asyncRefDetector->ProcessSDKOutput(output);
    faceInfoList = dynamic_cast<TNN_NS::UltraFaceDetectorOutput *>(output.get())->face_list;
    delete [] yuvData;
    delete [] rgbaData;
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
    LOGI("face info list size %d", faceInfoList.size());
    // TODO: copy face info list
    if (faceInfoList.size() > 0) {
        faceInfoArray = env->NewObjectArray(faceInfoList.size(), clsFaceInfo, NULL);
        for (int i = 0; i < faceInfoList.size(); i++) {
            jobject objFaceInfo = env->NewObject(clsFaceInfo, midconstructorFaceInfo);
            int landmarkNum = faceInfoList[i].key_points.size();
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
    TNN_NS::DimsVector target_dims = {1, 4, height, width};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, sourcePixelscolor);
    auto asyncRefDetector = gDetector;
    std::vector<TNN_NS::FaceInfo> faceInfoList;

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(input_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);
    AndroidBitmap_unlockPixels(env, imageSource);

    asyncRefDetector->ProcessSDKOutput(output);
    faceInfoList = dynamic_cast<TNN_NS::UltraFaceDetectorOutput *>(output.get())->face_list;

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
    LOGI("face info list size %d", faceInfoList.size());
    // TODO: copy face info list
    if (faceInfoList.size() > 0) {
        faceInfoArray = env->NewObjectArray(faceInfoList.size(), clsFaceInfo, NULL);
        for (int i = 0; i < faceInfoList.size(); i++) {
            jobject objFaceInfo = env->NewObject(clsFaceInfo, midconstructorFaceInfo);
            int landmarkNum = faceInfoList[i].key_points.size();
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
