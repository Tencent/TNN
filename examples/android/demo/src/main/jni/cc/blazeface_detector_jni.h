//
// Created by tencent on 2020-04-30.
//

#ifndef ANDROID_BLAZEFACE_DETECTOR_JNI_H
#define ANDROID_BLAZEFACE_DETECTOR_JNI_H
#include "BlazeFaceDetector.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>

#define TNN_BLAZEFACE_DETECTOR(sig) Java_com_tencent_tnn_demo_BlazeFaceDetector_##sig
#ifdef __cplusplus
extern "C"{
#endif

void makeBlazefaceDetectOption(std::shared_ptr<TNN_NS::BlazeFaceDetectorOption>& option, std::string& lib_path, std::string& proto_content, std:: string& model_content);
JNIEXPORT jint JNICALL TNN_BLAZEFACE_DETECTOR(init)(JNIEnv *env, jobject thiz, jstring modelPath,
                                                    jint width, jint height, jfloat scoreThreshold,
                                                    jfloat iouThreshold, jint topk,
                                                    jint computUnitType);
JNIEXPORT JNICALL jboolean TNN_BLAZEFACE_DETECTOR(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath);
JNIEXPORT JNICALL jint TNN_BLAZEFACE_DETECTOR(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jobjectArray TNN_BLAZEFACE_DETECTOR(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint rotate);
JNIEXPORT JNICALL jobjectArray TNN_BLAZEFACE_DETECTOR(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height);

#ifdef __cplusplus
}
#endif
#endif //ANDROID_BLAZEFACE_FACEDETECTOR_JNI_H
