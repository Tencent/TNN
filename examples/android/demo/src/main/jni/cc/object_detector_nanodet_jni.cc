//
// Created by tencent on 2020-04-29.
//
#include "object_detector_nanodet_jni.h"
#include "object_detector_nanodet.h"
#include "kannarotate.h"
#include "yuv420sp_to_rgb_fast_asm.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>
#include "tnn/utils/mat_utils.h"

static std::shared_ptr<TNN_NS::ObjectDetectorNanodet> gDetector;
static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu, 2 is huawei_npu
static jclass clsObjectInfo;
static jmethodID midconstructorObjectInfo;
static jfieldID fidx1;
static jfieldID fidy1;
static jfieldID fidx2;
static jfieldID fidy2;
static jfieldID fidscore;
static jfieldID fidlandmarks;
static jfieldID fidcls;
// Jni functions

JNIEXPORT JNICALL jint TNN_OBJECT_DETECTOR_NANODET(init)(JNIEnv *env, jobject thiz, jstring modelPath, jint computUnitType)
{
    // Reset bench description
    setBenchResult("");
    gDetector = std::make_shared<TNN_NS::ObjectDetectorNanodet>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/nanodet_m.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/nanodet_m.tnnmodel");
    // protoContent = fdLoadFile(modelPathStr + "/nanodet_e1.tnnproto");
    // modelContent = fdLoadFile(modelPathStr + "/nanodet_e1.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), modelContent.length());
    gComputeUnitType = computUnitType;

    TNN_NS::Status status = TNN_NS::TNN_OK;
    auto option = std::make_shared<TNN_NS::ObjectDetectorNanodetOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->library_path  = "";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    option->model_cfg     = "m";   // "m": nanodet_m; "e1": nanodet_efficientlite1
    LOGI("the device type  %d device huawei_npu" ,gComputeUnitType);
    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        status = gDetector->Init(option);
    } else if (gComputeUnitType == 2) {
        //add for huawei_npu store the om file
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
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

    if (clsObjectInfo == NULL)
    {
        clsObjectInfo = static_cast<jclass>(env->NewGlobalRef(env->FindClass("com/tencent/tnn/demo/ObjectInfo")));
        midconstructorObjectInfo = env->GetMethodID(clsObjectInfo, "<init>", "()V");
        fidx1 = env->GetFieldID(clsObjectInfo, "x1" , "F");
        fidy1 = env->GetFieldID(clsObjectInfo, "y1" , "F");
        fidx2 = env->GetFieldID(clsObjectInfo, "x2" , "F");
        fidy2 = env->GetFieldID(clsObjectInfo, "y2" , "F");
        fidscore = env->GetFieldID(clsObjectInfo, "score" , "F");
        fidlandmarks = env->GetFieldID(clsObjectInfo, "landmarks" , "[F");
        fidcls = env->GetFieldID(clsObjectInfo, "class_id", "I");
    }

    return 0;
}

JNIEXPORT JNICALL jboolean TNN_OBJECT_DETECTOR_NANODET(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
    TNN_NS::ObjectDetectorNanodet tmpDetector;
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/nanodet_m.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/nanodet_m.tnnmodel");
    // protoContent = fdLoadFile(modelPathStr + "/nanodet_e1.tnnproto");
    // modelContent = fdLoadFile(modelPathStr + "/nanodet_e1.tnnmodel");;
    auto option = std::make_shared<TNN_NS::ObjectDetectorNanodetOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
    option->library_path = "";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    option->model_cfg     = "m";   // "m": nanodet_m; "e1": nanodet_efficientlite1
    tmpDetector.setNpuModelPath(modelPathStr + "/");
    tmpDetector.setCheckNpuSwitch(true);
    TNN_NS::Status ret = tmpDetector.Init(option);
    return ret == TNN_NS::TNN_OK;
}

JNIEXPORT JNICALL jint TNN_OBJECT_DETECTOR_NANODET(deinit)(JNIEnv *env, jobject thiz)
{

    gDetector = nullptr;
    return 0;
}

//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"
JNIEXPORT JNICALL jobjectArray TNN_OBJECT_DETECTOR_NANODET(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint view_width, jint view_height, jint rotate)
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

    TNN_NS::DimsVector input_dims = {1, 4, width, height};

    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, input_dims, rgbaData);

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(input_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();

    TNN_NS::Status status = asyncRefDetector->Predict(input, output);

    objectInfoList = dynamic_cast<TNN_NS::ObjectDetectorNanodetOutput *>(output.get())->object_list;
    delete [] yuvData;
    delete [] rgbaData;
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int)status);
        return 0;
    }

    LOGI("object info list size %d", objectInfoList.size());
    // TODO: copy object info list
    if (objectInfoList.size() > 0) {
        objectInfoArray = env->NewObjectArray(objectInfoList.size(), clsObjectInfo, NULL);
        for (int i = 0; i < objectInfoList.size(); i++) {
            jobject objObjectInfo = env->NewObject(clsObjectInfo, midconstructorObjectInfo);
            int landmarkNum = objectInfoList[i].key_points.size();
            LOGI("object[%d] %f %f %f %f score %f landmark size %d, label_id: %d", i, objectInfoList[i].x1, objectInfoList[i].y1, objectInfoList[i].x2, objectInfoList[i].y2, objectInfoList[i].score, landmarkNum, objectInfoList[i].class_id);
            auto object_preview = objectInfoList[i].AdjustToImageSize(width, height);
            auto object_orig = object_preview.AdjustToViewSize(view_height, view_width, 2);
            env->SetFloatField(objObjectInfo, fidx1, object_orig.x1);
            env->SetFloatField(objObjectInfo, fidy1, object_orig.y1);
            env->SetFloatField(objObjectInfo, fidx2, object_orig.x2);
            env->SetFloatField(objObjectInfo, fidy2, object_orig.y2);
            env->SetFloatField(objObjectInfo, fidscore, object_orig.score);
            env->SetIntField(objObjectInfo, fidcls, object_orig.class_id);
            env->SetObjectArrayElement(objectInfoArray, i, objObjectInfo);
            env->DeleteLocalRef(objObjectInfo);
        }
        return objectInfoArray;
    } else {
        return 0;
    }

}
