//
// Created by tencent on 2020-04-30.
//

#ifndef ANDROID_IMAGECLASSIFY_JNI_H
#define ANDROID_IMAGECLASSIFY_JNI_H
#include <jni.h>
#define TNN_CLASSIFY(sig) Java_com_tencent_tnn_demo_ImageClassify_##sig
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT JNICALL jint TNN_CLASSIFY(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jint computeUnitType);
JNIEXPORT JNICALL jint TNN_CLASSIFY(deinit)(JNIEnv *env, jobject thiz);
JNIEXPORT JNICALL jintArray TNN_CLASSIFY(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height);

#ifdef __cplusplus
}
#endif
#endif //ANDROID_IMAGECLASSIFY_JNI_H
