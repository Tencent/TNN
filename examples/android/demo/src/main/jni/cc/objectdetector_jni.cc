//
// Created by tencent on 2020-04-29.
//
#include "objectdetector_jni.h"
#include "ObjectDetectorYolo.h"
#include "kannarotate.h"
#include "yuv420sp_to_rgb_fast_asm.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>

static std::shared_ptr<TNN_NS::ObjectDetectorYolo> gDetector;
static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu, 2 is npu
static jclass clsObjectInfo;
static jmethodID midconstructorObjectInfo;
static jfieldID fidx1;
static jfieldID fidy1;
static jfieldID fidx2;
static jfieldID fidy2;
static jfieldID fidscore;
static jfieldID fidlandmarks;
// Jni functions

JNIEXPORT JNICALL jint TNN_OBJECT_DETECTOR(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint width, jint height, jfloat scoreThreshold, jfloat iouThreshold, jint topk, jint computUnitType)
{
    // Reset bench description
    setBenchResult("");
    std::vector<int> nchw = {1, 3, height, width};
    gDetector = std::make_shared<TNN_NS::ObjectDetectorYolo>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/yolov5s-permute.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/yolov5s.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());
    gComputeUnitType = computUnitType;

    TNN_NS::Status status = TNN_NS::TNN_OK;
    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->library_path="";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    LOGE("the device type  %d device npu" ,gComputeUnitType);
    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        status = gDetector->Init(option);
    } else if (gComputeUnitType == 2) {
        //add for npu store the om file
        LOGE("the device type  %d device npu" ,gComputeUnitType);
        option->compute_units = TNN_NS::TNNComputeUnitsNPU;
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
    if (clsObjectInfo == NULL)
    {
        clsObjectInfo = static_cast<jclass>(env->NewGlobalRef(env->FindClass("com/tencent/tnn/demo/ObjectDetector$ObjectInfo")));
        midconstructorObjectInfo = env->GetMethodID(clsObjectInfo, "<init>", "()V");
        fidx1 = env->GetFieldID(clsObjectInfo, "x1" , "F");
        fidy1 = env->GetFieldID(clsObjectInfo, "y1" , "F");
        fidx2 = env->GetFieldID(clsObjectInfo, "x2" , "F");
        fidy2 = env->GetFieldID(clsObjectInfo, "y2" , "F");
        fidscore = env->GetFieldID(clsObjectInfo, "score" , "F");
        fidlandmarks = env->GetFieldID(clsObjectInfo, "landmarks" , "[F");
    }

    return 0;
}

JNIEXPORT JNICALL jboolean TNN_OBJECT_DETECTOR(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
    TNN_NS::ObjectDetectorYolo tmpDetector;
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/yolov5s-permute.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/yolov5s.tnnmodel");
    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->library_path = "";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    tmpDetector.setNpuModelPath(modelPathStr + "/");
    tmpDetector.setCheckNpuSwitch(true);
    TNN_NS::Status ret = tmpDetector.Init(option);
    return ret == TNN_NS::TNN_OK;
}

JNIEXPORT JNICALL jint TNN_OBJECT_DETECTOR(deinit)(JNIEnv *env, jobject thiz)
{

    gDetector = nullptr;
    return 0;
}

//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"
JNIEXPORT JNICALL jobjectArray TNN_OBJECT_DETECTOR(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint rotate)
{
    jobjectArray objectInfoArray;
    auto asyncRefDetector = gDetector;
    std::vector<TNN_NS::ObjectInfo> objectInfoList;
    // Convert yuv to rgb
    LOGI("detect from stream %d x %d r %d", width, height, rotate);
    unsigned char *yuvData = new unsigned char[height * width * 3 / 2];
    jbyte *yuvDataRef = env->GetByteArrayElements(yuv420sp, 0);
    int ret = kannarotate_yuv420sp((const unsigned char*)yuvDataRef, (int)width, (int)height, (unsigned char*)yuvData, (int)rotate);
    env->ReleaseByteArrayElements(yuv420sp, yuvDataRef, 0);
    unsigned char *rgbaData = new unsigned char[height * width * 4];
    yuv420sp_to_rgba_fast_asm((const unsigned char*)yuvData, height, width, (unsigned char*)rgbaData);
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector target_dims = {1, 3, height, width};

    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, rgbaData);
    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(input_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);

    asyncRefDetector->ProcessSDKOutput(output);
    objectInfoList = dynamic_cast<TNN_NS::ObjectDetectorYoloOutput *>(output.get())->object_list;
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
        device = "npu";
    }
    sprintf(temp, " device: %s \ntime:", device.c_str());
    std::string computeUnitTips(temp);
    std::string resultTips = std::string(computeUnitTips + asyncRefDetector->GetBenchResult().Description());
    setBenchResult(resultTips);
    LOGI("object info list size %d", objectInfoList.size());
    // TODO: copy object info list
    if (objectInfoList.size() > 0) {
        objectInfoArray = env->NewObjectArray(objectInfoList.size(), clsObjectInfo, NULL);
        for (int i = 0; i < objectInfoList.size(); i++) {
            jobject objObjectInfo = env->NewObject(clsObjectInfo, midconstructorObjectInfo);
            int landmarkNum = objectInfoList[i].key_points.size();
            LOGI("object[%d] %f %f %f %f score %f landmark size %d", i, objectInfoList[i].x1, objectInfoList[i].y1, objectInfoList[i].x2, objectInfoList[i].y2, objectInfoList[i].score, landmarkNum);
            env->SetFloatField(objObjectInfo, fidx1, objectInfoList[i].x1);
            env->SetFloatField(objObjectInfo, fidy1, objectInfoList[i].y1);
            env->SetFloatField(objObjectInfo, fidx2, objectInfoList[i].x2);
            env->SetFloatField(objObjectInfo, fidy2, objectInfoList[i].y2);
            env->SetFloatField(objObjectInfo, fidscore, objectInfoList[i].score);
            env->SetObjectArrayElement(objectInfoArray, i, objObjectInfo);
            env->DeleteLocalRef(objObjectInfo);
        }
        return objectInfoArray;
    } else {
        return 0;
    }

}

JNIEXPORT JNICALL jobjectArray TNN_OBJECT_DETECTOR(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width, jint height)
{
    jobjectArray objectInfoArray;
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
    TNN_NS::DimsVector target_dims = {1, 3, height, width};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, sourcePixelscolor);
    auto asyncRefDetector = gDetector;
    std::vector<TNN_NS::ObjectInfo> objectInfoList;

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(input_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);
    AndroidBitmap_unlockPixels(env, imageSource);

    asyncRefDetector->ProcessSDKOutput(output);
    objectInfoList = dynamic_cast<TNN_NS::ObjectDetectorYoloOutput *>(output.get())->object_list;

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
        device = "npu";
    }
    sprintf(temp, " device: %s \ntime:", device.c_str());
    std::string computeUnitTips(temp);
    std::string resultTips = std::string(computeUnitTips + asyncRefDetector->GetBenchResult().Description());
    setBenchResult(resultTips);
    LOGI("object info list size %d", objectInfoList.size());
    // TODO: copy object info list
    if (objectInfoList.size() > 0) {
        objectInfoArray = env->NewObjectArray(objectInfoList.size(), clsObjectInfo, NULL);
        for (int i = 0; i < objectInfoList.size(); i++) {
            jobject objObjectInfo = env->NewObject(clsObjectInfo, midconstructorObjectInfo);
            int landmarkNum = objectInfoList[i].key_points.size();
            LOGI("object[%d] %f %f %f %f score %f landmark size %d", i, objectInfoList[i].x1, objectInfoList[i].y1, objectInfoList[i].x2, objectInfoList[i].y2, objectInfoList[i].score, landmarkNum);
            env->SetFloatField(objObjectInfo, fidx1, objectInfoList[i].x1);
            env->SetFloatField(objObjectInfo, fidy1, objectInfoList[i].y1);
            env->SetFloatField(objObjectInfo, fidx2, objectInfoList[i].x2);
            env->SetFloatField(objObjectInfo, fidy2, objectInfoList[i].y2);
            env->SetFloatField(objObjectInfo, fidscore, objectInfoList[i].score);
            env->SetObjectArrayElement(objectInfoArray, i, objObjectInfo);
            env->DeleteLocalRef(objObjectInfo);
        }
        return objectInfoArray;
    } else {
        return 0;
    }

    return 0;
}
