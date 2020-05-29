//
// Created by tencent on 2020-04-30.
//

#ifndef ANDROID_FACEDETECTOR_JNI_H
#define ANDROID_FACEDETECTOR_JNI_H
#include <jni.h>
#define TNN_FACE_DETECTOR(sig) Java_com_tencent_tnn_demo_FaceDetector_##sig
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT JNICALL jint TNN_FACE_DETECTOR(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jfloat scoreThreshold, jfloat iouThreshold, jint topk, jint computUnitType);
JNIEXPORT JNICALL jint TNN_FACE_DETECTOR(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jobjectArray TNN_FACE_DETECTOR(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint rotate);
JNIEXPORT JNICALL jobjectArray TNN_FACE_DETECTOR(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height);

#ifdef __cplusplus
}
#endif
#endif //ANDROID_FACEDETECTOR_JNI_H
