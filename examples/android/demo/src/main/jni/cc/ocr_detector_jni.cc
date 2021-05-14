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

#include "ocr_detector_jni.h"
#if HAS_OPENCV
#include "ocr_driver.h"
#include "ocr_textbox_detector.h"
#include "ocr_text_recognizer.h"
#include "ocr_angle_predictor.h"
#include "kannarotate.h"
#include "yuv420sp_to_rgb_fast_asm.h"
#endif
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>

#if HAS_OPENCV
static std::shared_ptr<TNN_NS::OCRDriver> gOCRDriver;
static std::shared_ptr<TNN_NS::OCRTextboxDetector> gOCRTextboxDetector;
static std::shared_ptr<TNN_NS::OCRAnglePredictor> gOCRAnglePredictor;
static std::shared_ptr<TNN_NS::OCRTextRecognizer> gOCRTextRecognizer;
#endif
static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu, 2 is huawei_npu
static jclass clsObjectInfo;
static jmethodID midconstructorObjectInfo;
static jfieldID fidkeypoints;
static jfieldID fidlabel;
// Jni functions

JNIEXPORT JNICALL jint TNN_OCR_DETECTOR(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jint computUnitType)
{
    // Reset bench description
    setBenchResult("");
#if HAS_OPENCV
    std::vector<int> nchw = {1, 3, height, width};
    gOCRDriver = std::make_shared<TNN_NS::OCRDriver>();
    gOCRTextboxDetector = std::make_shared<TNN_NS::OCRTextboxDetector>();
    gOCRAnglePredictor = std::make_shared<TNN_NS::OCRAnglePredictor>();
    gOCRTextRecognizer = std::make_shared<TNN_NS::OCRTextRecognizer>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/dbnet.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/dbnet.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());
    gComputeUnitType = computUnitType;

    TNN_NS::Status status = TNN_NS::TNN_OK;
    {
        auto option = std::make_shared<TNN_NS::OCRTextboxDetectorOption>();
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        option->library_path="";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        option->scale_down_ratio = 0.75f;
        option->padding = 10;
        if (gComputeUnitType == 1) {
            option->compute_units = TNN_NS::TNNComputeUnitsGPU;
            status = gOCRTextboxDetector->Init(option);
        } else if (gComputeUnitType == 2) {
            //add for huawei_npu store the om file
            option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
            gOCRTextboxDetector->setNpuModelPath(modelPathStr + "/");
            gOCRTextboxDetector->setCheckNpuSwitch(false);
            status = gOCRTextboxDetector->Init(option);
        } else {
            option->compute_units = TNN_NS::TNNComputeUnitsCPU;
            status = gOCRTextboxDetector->Init(option);
        }

        if (status != TNN_NS::TNN_OK) {
            LOGE("ocr textbox detector init failed %d", (int)status);
            return -1;
        }
    }

    protoContent = fdLoadFile(modelPathStr + "/angle_net.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/angle_net.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());

    status = TNN_NS::TNN_OK;
    {
        auto option = std::make_shared<TNN_NS::TNNSDKOption>();
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        option->library_path="";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        if (gComputeUnitType == 1) {
            option->compute_units = TNN_NS::TNNComputeUnitsGPU;
            status = gOCRAnglePredictor->Init(option);
        } else if (gComputeUnitType == 2) {
            //add for huawei_npu store the om file
            option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
            gOCRAnglePredictor->setNpuModelPath(modelPathStr + "/");
            gOCRAnglePredictor->setCheckNpuSwitch(false);
            status = gOCRAnglePredictor->Init(option);
        } else {
            option->compute_units = TNN_NS::TNNComputeUnitsCPU;
            status = gOCRAnglePredictor->Init(option);
        }

        if (status != TNN_NS::TNN_OK) {
            LOGE("ocr angle predictor init failed %d", (int)status);
            return -1;
        }
    }

    protoContent = fdLoadFile(modelPathStr + "/crnn_lite_lstm.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/crnn_lite_lstm.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());

    status = TNN_NS::TNN_OK;
    {
        auto recognizer_option = std::make_shared<TNN_NS::OCRTextRecognizerOption>();
        recognizer_option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        recognizer_option->library_path="";
        recognizer_option->vocab_path=modelPathStr + "/keys.txt";
        recognizer_option->proto_content = protoContent;
        recognizer_option->model_content = modelContent;
        if (gComputeUnitType == 1) {
            recognizer_option->compute_units = TNN_NS::TNNComputeUnitsGPU;
            status = gOCRTextRecognizer->Init(recognizer_option);
        } else if (gComputeUnitType == 2) {
            //add for huawei_npu store the om file
            recognizer_option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
            gOCRTextRecognizer->setNpuModelPath(modelPathStr + "/");
            gOCRTextRecognizer->setCheckNpuSwitch(false);
            status = gOCRTextRecognizer->Init(recognizer_option);
        } else {
            recognizer_option->compute_units = TNN_NS::TNNComputeUnitsCPU;
            status = gOCRTextRecognizer->Init(recognizer_option);
        }

        if (status != TNN_NS::TNN_OK) {
            LOGE("ocr text recognizer init failed %d", (int)status);
            return -1;
        }
    }

    status = gOCRDriver->Init({gOCRTextboxDetector, gOCRAnglePredictor, gOCRTextRecognizer});
    if (status != TNN_NS::TNN_OK) {
        LOGE("ocr detector init failed %d", (int)status);
        return -1;
    }

    if (clsObjectInfo == NULL) {
        clsObjectInfo = static_cast<jclass>(env->NewGlobalRef(env->FindClass("com/tencent/tnn/demo/ObjectInfo")));
        midconstructorObjectInfo = env->GetMethodID(clsObjectInfo, "<init>", "()V");
        fidkeypoints = env->GetFieldID(clsObjectInfo, "key_points", "[[F");
        fidlabel = env->GetFieldID(clsObjectInfo, "label" , "Ljava/lang/String;");
    }
#endif

    return 0;
}

JNIEXPORT JNICALL jboolean TNN_OCR_DETECTOR(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
#if HAS_OPENCV
    // ocr detector relys on the support of Reshape which is not supported on NPU network for now
    return false;
    std::shared_ptr<TNN_NS::OCRDriver> tmpOCRDriver = std::make_shared<TNN_NS::OCRDriver>();
    std::shared_ptr<TNN_NS::OCRTextboxDetector> tmpOCRTextboxDetector = std::make_shared<TNN_NS::OCRTextboxDetector>();
    std::shared_ptr<TNN_NS::OCRAnglePredictor> tmpOCRAnglePredictor = std::make_shared<TNN_NS::OCRAnglePredictor>();
    std::shared_ptr<TNN_NS::OCRTextRecognizer> tmpOCRTextRecognizer = std::make_shared<TNN_NS::OCRTextRecognizer>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/dbnet.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/dbnet.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());

    TNN_NS::Status status = TNN_NS::TNN_OK;
    {
        auto option = std::make_shared<TNN_NS::OCRTextboxDetectorOption>();
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        option->library_path="";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        //add for huawei_npu store the om file
        tmpOCRTextboxDetector->setNpuModelPath(modelPathStr + "/");
        tmpOCRTextboxDetector->setCheckNpuSwitch(false);
        status = tmpOCRTextboxDetector->Init(option);

        if (status != TNN_NS::TNN_OK) {
            LOGE("ocr textbox detector init failed %d", (int)status);
            return false;
        }
    }

    protoContent = fdLoadFile(modelPathStr + "/angle_net.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/angle_net.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());

    status = TNN_NS::TNN_OK;
    {
        auto option = std::make_shared<TNN_NS::TNNSDKOption>();
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        option->library_path="";
        option->proto_content = protoContent;
        option->model_content = modelContent;
        tmpOCRAnglePredictor->setNpuModelPath(modelPathStr + "/");
        tmpOCRAnglePredictor->setCheckNpuSwitch(false);
        status = tmpOCRAnglePredictor->Init(option);

        if (status != TNN_NS::TNN_OK) {
            LOGE("ocr angle predictor init failed %d", (int)status);
            return false;
        }
    }

    protoContent = fdLoadFile(modelPathStr + "/crnn_lite_lstm.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/crnn_lite_lstm.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());

    status = TNN_NS::TNN_OK;
    {
        auto recognizer_option = std::make_shared<TNN_NS::OCRTextRecognizerOption>();
        recognizer_option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        recognizer_option->library_path="";
        recognizer_option->vocab_path=modelPathStr + "/keys.txt";
        recognizer_option->proto_content = protoContent;
        recognizer_option->model_content = modelContent;
        //add for huawei_npu store the om file
        tmpOCRTextRecognizer->setNpuModelPath(modelPathStr + "/");
        tmpOCRTextRecognizer->setCheckNpuSwitch(false);
        status = tmpOCRTextRecognizer->Init(recognizer_option);

        if (status != TNN_NS::TNN_OK) {
            LOGE("ocr text recognizer init failed %d", (int)status);
            return false;
        }
    }

    status = tmpOCRDriver->Init({tmpOCRTextboxDetector, tmpOCRAnglePredictor, tmpOCRTextRecognizer});
    if (status != TNN_NS::TNN_OK) {
        LOGE("ocr detector init failed %d", (int)status);
        return false;
    }

    return true;
#else
    return false;
#endif
}

JNIEXPORT JNICALL jint TNN_OCR_DETECTOR(deinit)(JNIEnv *env, jobject thiz)
{

#if HAS_OPENCV
    gOCRTextboxDetector = nullptr;
    gOCRAnglePredictor = nullptr;
    gOCRTextRecognizer = nullptr;
    gOCRDriver = nullptr;
#endif
    return 0;
}

JNIEXPORT JNICALL jobjectArray TNN_OCR_DETECTOR(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint view_width, jint view_height, jint rotate)
{
#if HAS_OPENCV
    jobjectArray objectInfoArray;
    auto asyncRefDetector = gOCRDriver;
    TNN_NS::OCROutput* ocrOutput;
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

    asyncRefDetector->ProcessSDKOutput(output);
    delete [] yuvData;
    delete [] rgbaData;
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int)status);
        return 0;
    }

    ocrOutput = dynamic_cast<TNN_NS::OCROutput *>(output.get());
    if (ocrOutput && ocrOutput->texts.size() > 0) {
        objectInfoArray = env->NewObjectArray(ocrOutput->texts.size(), clsObjectInfo, NULL);
        for (int i = 0; i < ocrOutput->texts.size(); i++) {
            jobject objObjectInfo = env->NewObject(clsObjectInfo, midconstructorObjectInfo);

            TNN_NS::ObjectInfo objectInfo;
            objectInfo.key_points.push_back(ocrOutput->box[i * 4]);
            objectInfo.key_points.push_back(ocrOutput->box[i * 4 + 1]);
            objectInfo.key_points.push_back(ocrOutput->box[i * 4 + 2]);
            objectInfo.key_points.push_back(ocrOutput->box[i * 4 + 3]);
            objectInfo.image_width = ocrOutput->image_width;
            objectInfo.image_height = ocrOutput->image_height;
            objectInfo.label = ocrOutput->texts[i].c_str();

            auto object_preview = objectInfo.AdjustToImageSize(width, height);
            auto object_orig = object_preview.AdjustToViewSize(view_height, view_width, 2);

            jclass cls1dArr = env->FindClass("[F");
            // Create the returnable jobjectArray with an initial value
            jobjectArray outer = env->NewObjectArray(objectInfo.key_points.size(), cls1dArr, NULL);
            for (int j = 0; j < objectInfo.key_points.size(); j++) {
                jfloatArray inner = env->NewFloatArray(2);
                float temp[] = {object_orig.key_points[j].first, object_orig.key_points[j].second};
                env->SetFloatArrayRegion(inner, 0, 2, temp);
                env->SetObjectArrayElement(outer, j, inner);
                env->DeleteLocalRef(inner);
            }
            env->SetObjectField(objObjectInfo, fidkeypoints, outer);
            jstring str = env->NewStringUTF(objectInfo.label);
            env->SetObjectField(objObjectInfo, fidlabel, str);
            env->DeleteLocalRef(str);
            env->SetObjectArrayElement(objectInfoArray, i, objObjectInfo);
            env->DeleteLocalRef(objObjectInfo);
        }
        return objectInfoArray;
    } else {
        return 0;
    }
#endif
    return 0;
}

JNIEXPORT JNICALL jobjectArray TNN_OCR_DETECTOR(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height)
{
#if HAS_OPENCV
    jobjectArray objectInfoArray;
    TNN_NS::OCROutput* ocrOutput;
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
    gOCRDriver->SetBenchOption(bench_option);
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector target_dims = {1, 4, height, width};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, sourcePixelscolor);
    auto asyncRefDetector = gOCRDriver;

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(input_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int)status);
        return 0;
    }

    char temp[128] = "";
    std::string device = "arm";
    if (gComputeUnitType == 1) {
        device = "gpu";
    } else if (gComputeUnitType == 2) {
        device = "huawei_npu";
    }
    sprintf(temp, " device: %s \n", device.c_str());
    std::string computeUnitTips(temp);
    setBenchResult(computeUnitTips);

    asyncRefDetector->ProcessSDKOutput(output);
    AndroidBitmap_unlockPixels(env, imageSource);
    ocrOutput = dynamic_cast<TNN_NS::OCROutput *>(output.get());

    if (ocrOutput && ocrOutput->texts.size() > 0) {
        objectInfoArray = env->NewObjectArray(ocrOutput->texts.size(), clsObjectInfo, NULL);
        for (int i = 0; i < ocrOutput->texts.size(); i++) {
            jobject objObjectInfo = env->NewObject(clsObjectInfo, midconstructorObjectInfo);

            TNN_NS::ObjectInfo objectInfo;
            objectInfo.key_points.push_back(ocrOutput->box[i * 4]);
            objectInfo.key_points.push_back(ocrOutput->box[i * 4 + 1]);
            objectInfo.key_points.push_back(ocrOutput->box[i * 4 + 2]);
            objectInfo.key_points.push_back(ocrOutput->box[i * 4 + 3]);
            objectInfo.image_width = ocrOutput->image_width;
            objectInfo.image_height = ocrOutput->image_height;
            objectInfo.label = ocrOutput->texts[i].c_str();

            jclass cls1dArr = env->FindClass("[F");
            // Create the returnable jobjectArray with an initial value
            jobjectArray outer = env->NewObjectArray(objectInfo.key_points.size(), cls1dArr, NULL);
            for (int j = 0; j < objectInfo.key_points.size(); j++) {
                jfloatArray inner = env->NewFloatArray(2);
                float temp[] = {objectInfo.key_points[j].first, objectInfo.key_points[j].second};
                env->SetFloatArrayRegion(inner, 0, 2, temp);
                env->SetObjectArrayElement(outer, j, inner);
                env->DeleteLocalRef(inner);
            }
            env->SetObjectField(objObjectInfo, fidkeypoints, outer);
            jstring str = env->NewStringUTF(objectInfo.label);
            env->SetObjectField(objObjectInfo, fidlabel, str);
            env->DeleteLocalRef(str);
            env->SetObjectArrayElement(objectInfoArray, i, objObjectInfo);
            env->DeleteLocalRef(objObjectInfo);
        }
        return objectInfoArray;
    } else {
        return 0;
    }
#endif
    return 0;
}
