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

static std::shared_ptr<TNN_NS::ImageClassifier> gDetector;
static int gComputeUnitType = 0;

JNIEXPORT JNICALL jint TNN_CLASSIFY(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jint computeUnitType)
{
    // Reset bench description
    setBenchResult("");
    std::vector<int> nchw = {1, 3, height, width};
    gDetector = std::make_shared<TNN_NS::ImageClassifier>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/squeezenet_v1.1.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/squeezenet_v1.1.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());
    TNN_NS::Status status = TNN_NS::TNN_OK;
    gComputeUnitType = computeUnitType;

    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->input_shapes = {};
    option->library_path="";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
    } else if (gComputeUnitType == 2) {
        LOGI("the device type  %d device huawei_npu" ,gComputeUnitType);
        gDetector->setNpuModelPath(modelPathStr + "/");
        gDetector->setCheckNpuSwitch(false);
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
    } else {
	    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    }
    status = gDetector->Init(option);

    if (status != TNN_NS::TNN_OK) {
        LOGE("detector init failed %d", (int)status);
        return -1;
    }
    TNN_NS::BenchOption bench_option;
    bench_option.forward_count = 20;
    gDetector->SetBenchOption(bench_option);
    return 0;
}

JNIEXPORT jboolean TNN_CLASSIFY(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
    TNN_NS::ImageClassifier tmpDetector;
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/squeezenet_v1.1.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/squeezenet_v1.1.tnnmodel");

    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
    option->input_shapes = {};
    option->library_path="";
    option->proto_content = protoContent;
    option->model_content = modelContent;

    tmpDetector.setNpuModelPath(modelPathStr + "/");
    tmpDetector.setCheckNpuSwitch(true);
    TNN_NS::Status ret = tmpDetector.Init(option);
    return ret == TNN_NS::TNN_OK;
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
    std::shared_ptr<TNN_NS::Mat> input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, sourcePixelscolor);
    int resultList[1];

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(input_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = gDetector->CreateSDKOutput();
    TNN_NS::Status status = gDetector->Predict(input, output);
    //get output map
    gDetector->ProcessSDKOutput(output);

    AndroidBitmap_unlockPixels(env, imageSource);

    if (status != TNN_NS::TNN_OK) {
        return 0;
    }
    char temp[128] = "";
    std::string device = "arm";
    if (gComputeUnitType == 1) {
        device = "gpu";
    } else if (gComputeUnitType == 2) {
        device = "huawei_npu";
    }
    sprintf(temp, " device: %s \ntime: ", device.c_str());
    std::string computeUnitTips(temp);
    std::string resultTips = std::string(computeUnitTips + gDetector->GetBenchResult().Description());
    setBenchResult(resultTips);
    LOGE("classify id %d", resultList[0]);
    resultArray = env->NewIntArray(1);
    resultList[0] =  dynamic_cast<TNN_NS::ImageClassifierOutput*>(output.get())->class_id;
    env->SetIntArrayRegion(resultArray, 0, 1, resultList);

    return resultArray;
}
