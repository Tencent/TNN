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

#include "hair_segmentation_jni.h"
#include "hair_segmentation.h"
#include "tnn_sdk_sample.h"
#include "kannarotate.h"
#include "yuv420sp_to_rgb_fast_asm.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>

static std::shared_ptr<TNN_NS::HairSegmentation> gSegmentator;
static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu, 2 is huawei_npu
static jclass clsImageInfo;
static jmethodID midconstructorImageInfo;
static jfieldID fidimage_width;
static jfieldID fidimage_height;
static jfieldID fidimage_channel;
static jfieldID fiddata;
// Jni functions

JNIEXPORT JNICALL jint TNN_HAIR_SEGMENTATION(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jint computeUnitType) {
    // Reset bench description
    setBenchResult("");
    gSegmentator = std::make_shared<TNN_NS::HairSegmentation>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/segmentation.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/segmentation.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());
    gComputeUnitType = computeUnitType;

    TNN_NS::Status status = TNN_NS::TNN_OK;
    auto option = std::make_shared<TNN_NS::HairSegmentationOption>();
    option->library_path = "";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    option->input_width = 256;
    option->input_height = 256;
    option->mode = 1;
    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        status = gSegmentator->Init(option);
    } else if (gComputeUnitType == 2) {
        //add for huawei_npu store the om file
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        gSegmentator->setNpuModelPath(modelPathStr + "/");
        gSegmentator->setCheckNpuSwitch(false);
        status = gSegmentator->Init(option);
    } else {
	    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    	status = gSegmentator->Init(option);
    }

    if (status != TNN_NS::TNN_OK) {
        LOGE("detector init failed %d", (int)status);
        return -1;
    }

    if (clsImageInfo == NULL) {
        clsImageInfo = static_cast<jclass>(env->NewGlobalRef(env->FindClass("com/tencent/tnn/demo/ImageInfo")));
        midconstructorImageInfo = env->GetMethodID(clsImageInfo, "<init>", "()V");
        fidimage_width = env->GetFieldID(clsImageInfo, "image_width" , "I");
        fidimage_height = env->GetFieldID(clsImageInfo, "image_height" , "I");
        fidimage_channel = env->GetFieldID(clsImageInfo, "image_channel" , "I");
        fiddata = env->GetFieldID(clsImageInfo, "data" , "[B");
    }

    return 0;
}

JNIEXPORT JNICALL jboolean TNN_HAIR_SEGMENTATION(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
    TNN_NS::HairSegmentation tmpSegmentator;
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/segmentation.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/segmentation.tnnmodel");
    auto option = std::make_shared<TNN_NS::HairSegmentationOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
    option->library_path = "";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    option->input_height= 256;
    option->input_width = 256;
    tmpSegmentator.setNpuModelPath(modelPathStr + "/");
    tmpSegmentator.setCheckNpuSwitch(true);
    TNN_NS::Status ret = tmpSegmentator.Init(option);
    return ret == TNN_NS::TNN_OK;
}

JNIEXPORT JNICALL jint TNN_HAIR_SEGMENTATION(deinit)(JNIEnv *env, jobject thiz) {
    gSegmentator = nullptr;
    return 0;
}

JNIEXPORT JNICALL jint TNN_HAIR_SEGMENTATION(setHairColor)(JNIEnv *env, jobject thiz, jbyteArray rgba) {
    const unsigned char *rgbaDataRef = (const unsigned char*)env->GetByteArrayElements(rgba, 0);
    TNN_NS::RGBA colorData(rgbaDataRef[0], rgbaDataRef[1], rgbaDataRef[2], rgbaDataRef[3]);
    gSegmentator->SetHairColor(colorData);

    return 0;
}

JNIEXPORT JNICALL jobjectArray TNN_HAIR_SEGMENTATION(predictFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint rotate)
{
    jobjectArray imageInfoArray;
    auto asyncRefSegmentator = gSegmentator;
    std::vector<TNN_NS::ImageInfo> imageInfoList;
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
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = asyncRefSegmentator->CreateSDKOutput();

    TNN_NS::Status status = asyncRefSegmentator->Predict(input, output);

    asyncRefSegmentator->ProcessSDKOutput(output);
    imageInfoList.push_back(dynamic_cast<TNN_NS::HairSegmentationOutput *>(output.get())->hair_mask);
    imageInfoList.push_back(dynamic_cast<TNN_NS::HairSegmentationOutput *>(output.get())->merged_image);
    delete [] yuvData;
    delete [] rgbaData;
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int)status);
        return 0;
    }

    if (imageInfoList.size() > 0) {
        imageInfoArray = env->NewObjectArray(imageInfoList.size(), clsImageInfo, NULL);
        for (int i = 0; i < imageInfoList.size(); i++) {
            jobject objImageInfo = env->NewObject(clsImageInfo, midconstructorImageInfo);
            int image_width = imageInfoList[i].image_width;
            int image_height = imageInfoList[i].image_height;
            int image_channel = imageInfoList[i].image_channel;
            int dataNum = image_channel * image_width * image_height;

            env->SetIntField(objImageInfo, fidimage_width, image_width);
            env->SetIntField(objImageInfo, fidimage_height, image_height);
            env->SetIntField(objImageInfo, fidimage_channel, image_channel);

            jbyteArray jarrayData = env->NewByteArray(dataNum);
            env->SetByteArrayRegion(jarrayData, 0, dataNum , (jbyte*)imageInfoList[i].data.get());
            env->SetObjectField(objImageInfo, fiddata, jarrayData);

            env->SetObjectArrayElement(imageInfoArray, i, objImageInfo);
            env->DeleteLocalRef(objImageInfo);
        }
        return imageInfoArray;
    } else {
        return 0;
    }

}