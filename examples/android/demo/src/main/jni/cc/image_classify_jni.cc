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

#include "image_classifier.h"
#include "image_classify_jni.h"
#include "helper_jni.h"
#include <android/bitmap.h>

static std::shared_ptr<ImageClassifier> gDetector;
static int gComputeUnitType = 0;

JNIEXPORT JNICALL jint TNN_CLASSIFY(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jint computeUnitType)
{
    // Reset bench description
    setBenchResult("");
    std::vector<int> nchw = {1, 3, height, width};
    gDetector = std::make_shared<ImageClassifier>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/squeezenet_v1.1.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/squeezenet_v1.1.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());
    TNN_NS::Status status;
    gComputeUnitType = computeUnitType;
    if (gComputeUnitType == 0) {
        status = gDetector->Init(protoContent, modelContent, "", TNN_NS::TNNComputeUnitsCPU);
    } else {
        status = gDetector->Init(protoContent, modelContent, "", TNN_NS::TNNComputeUnitsGPU);
    }

    if (status != TNN_NS::TNN_OK) {
        LOGE("detector init failed %d", (int)status);
        return -1;
    }
    TNN_NS::BenchOption bench_option;
    bench_option.forward_count = 20;
    gDetector->SetBenchOption(bench_option);
    return 0;
}
JNIEXPORT JNICALL jint TNN_CLASSIFY(deinit)(JNIEnv *env, jobject thiz)
{

    gDetector = nullptr;
    return 0;
}
JNIEXPORT JNICALL jintArray TNN_CLASSIFY(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height)
{
    jintArray resultArray;
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
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector target_dims = {1, 3, height, width};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, sourcePixelscolor);
    int resultList[1];
    TNN_NS::Status status = gDetector->Classify(input_mat, width, height, resultList[0]);
    AndroidBitmap_unlockPixels(env, imageSource);

    if (status != TNN_NS::TNN_OK) {
        return 0;
    }
    char temp[128] = "";
    sprintf(temp, " device: %s \ntime: ", (gComputeUnitType==0)?"arm":"gpu");
    std::string computeUnitTips(temp);
    std::string resultTips = std::string(computeUnitTips + gDetector->GetBenchResult().Description());
    setBenchResult(resultTips);
    LOGE("classify id %d", resultList[0]);
    resultArray = env->NewIntArray(1);
    env->SetIntArrayRegion(resultArray, 0, 1, resultList);

    return resultArray;
}
