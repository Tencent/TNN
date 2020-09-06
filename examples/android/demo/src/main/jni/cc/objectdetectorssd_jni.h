//
// Created by tencent on 2020-04-30.
//

#ifndef ANDROID_OBJECTDETECTORSSD_JNI_H
#define ANDROID_OBJECTDETECTORSSD_JNI_H
#include "jni.h"
#define TNN_OBJECT_DETECTORSSD(sig) Java_com_tencent_tnn_demo_ObjectDetectorSSD_##sig
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT JNICALL jint TNN_OBJECT_DETECTORSSD(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jfloat scoreThreshold, jfloat iouThreshold, jint topk, jint computUnitType);
JNIEXPORT JNICALL jboolean TNN_OBJECT_DETECTORSSD(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath);
JNIEXPORT JNICALL jint TNN_OBJECT_DETECTORSSD(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jobjectArray TNN_OBJECT_DETECTORSSD(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint rotate);
JNIEXPORT JNICALL jobjectArray TNN_OBJECT_DETECTORSSD(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height);

#ifdef __cplusplus
}
#endif
#endif //ANDROID_OBJECTDETECTORSSD_JNI_H
