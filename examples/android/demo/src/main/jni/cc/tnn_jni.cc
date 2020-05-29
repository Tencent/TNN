#include "tnn_jni.h"
#include "tnn_lib.h"
#include <android/bitmap.h>
#include <chrono>
#include<stdlib.h>

using namespace std::chrono;

char * global_path = "/storage/emulated/0/log";

static jfieldID getHandleField(JNIEnv *env, jobject obj){
    jclass c = env->GetObjectClass(obj);
    // J is the type signature for long:
    return env->GetFieldID(c, "nativePtr", "J");
}

template <typename T>
T *getHandle(JNIEnv *env, jobject obj){
    jlong handle = env->GetLongField(obj, getHandleField(env, obj));
    return reinterpret_cast<T *>(handle);
}

template <typename T>
void setHandle(JNIEnv *env, jobject obj, T *t){
    jlong handle = reinterpret_cast<jlong>(t);
    env->SetLongField(obj, getHandleField(env, obj), handle);
}


char* jstringTostring(JNIEnv* env, jstring jstr)
{
    char* rtn = NULL;
    jclass clsstring = env->FindClass("java/lang/String");
    jstring strencode = env->NewStringUTF("utf-8");
    jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
    jbyteArray barr= (jbyteArray)env->CallObjectMethod(jstr, mid, strencode);
    jsize alen = env->GetArrayLength(barr);
    jbyte* ba = env->GetByteArrayElements(barr, JNI_FALSE);
    if (alen > 0)
    {
        rtn = (char*)malloc(alen + 1);
        memcpy(rtn, ba, alen);
        rtn[alen] = 0;
    }
    env->ReleaseByteArrayElements(barr, ba, 0);
    return rtn;
}

jstring stoJstring(JNIEnv* env, const char* pat)
{
    jclass strClass = env->FindClass("Ljava/lang/String;");
    jmethodID ctorID = env->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");
    jbyteArray bytes = env->NewByteArray(strlen(pat));
    env->SetByteArrayRegion(bytes, 0, strlen(pat), (jbyte*)pat);
    jstring encoding = env->NewStringUTF("utf-8");
    return (jstring)env->NewObject(strClass, ctorID, bytes, encoding);
}


JNIEXPORT jint JNICALL Java_com_tencent_tnn_demo_TNNLib_init(JNIEnv* env, jobject thiz, jstring protoFilePath, jstring modelFilePath, jstring device) {
    TNNLib * lib = new TNNLib();
    int result = lib->Init(jstringTostring(env, protoFilePath), jstringTostring(env, modelFilePath), jstringTostring(env, device));
    setHandle(env, thiz, lib);
    return result;
}

JNIEXPORT jfloatArray JNICALL Java_com_tencent_tnn_demo_TNNLib_forward(JNIEnv* env, jobject thiz, jobject imageSource) {
    TNNLib* inst = getHandle<TNNLib>(env, thiz);

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
    std::vector<float> results = inst->Forward(sourcePixelscolor);

    jfloatArray result_array;
    result_array = env->NewFloatArray(results.size());
    env->SetFloatArrayRegion(result_array, 0, results.size(), results.data());

    AndroidBitmap_unlockPixels(env, imageSource);

    return result_array;
}

JNIEXPORT jint JNICALL Java_com_tencent_tnn_demo_TNNLib_deinit(
        JNIEnv *env, jobject thiz) {
    TNNLib* inst = getHandle<TNNLib>(env, thiz);
    delete inst;
    setHandle(env, thiz, (TNNLib*)0);
    return 0;
}



