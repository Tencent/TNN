#include "blazeface_detector_jni.h"
#include "blazeface_detector.h"
#include <jni.h>
#include "helper_jni.h"
#include <android/bitmap.h>
#include <tnn/utils/mat_utils.h>
#include <kannarotate-android-lib/include/kannarotate.h>
#include <yuv420sp_to_rgb_fast_asm.h>

static std::shared_ptr<TNN_NS::BlazeFaceDetector> gDetector;
static jclass clsFaceInfo;
static int gComputeUnitType = 0; // 0 is cpu, 1 is gpu, 2 is huawei_npu
static int target_width = 128;
static int target_height = 128;

static std::string modelPathStr = "";
static jmethodID midconstructorFaceInfo;
static jfieldID fidx1;
static jfieldID fidy1;
static jfieldID fidx2;
static jfieldID fidy2;
static jfieldID fidkeypoints;

JNIEXPORT jint JNICALL TNN_BLAZEFACE_DETECTOR(init)(JNIEnv *env, jobject thiz, jstring modelPath,
                                                    jint width, jint height, jfloat scoreThreshold,
                                                    jfloat iouThreshold, jint topk,
                                                    jint computUnitType) {
    // Reset bench description
    LOGE("image height width %d %d \n", height, width);
    gDetector = std::make_shared<TNN_NS::BlazeFaceDetector>();
    std::string proto_content, model_content, lib_path = "";
    modelPathStr = jstring2string(env, modelPath);
    proto_content = fdLoadFile(modelPathStr + "/blazeface.tnnproto");
    model_content = fdLoadFile(modelPathStr + "/blazeface.tnnmodel");
    LOGI("proto content size %d model content size %d", proto_content.length(),
         model_content.length());
    gComputeUnitType = computUnitType;

    TNN_NS::Status status = TNN_NS::TNN_OK;
    auto option = std::make_shared<TNN_NS::BlazeFaceDetectorOption>();
    makeBlazefaceDetectOption(option, lib_path, proto_content, model_content);

    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        status = gDetector->Init(option);
    } else if (gComputeUnitType == 2) {
        //add for huawei_npu store the om file
        LOGE("the device type  %d device huawei_npu", gComputeUnitType);
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
        gDetector->setNpuModelPath(modelPathStr + "/");
        gDetector->setCheckNpuSwitch(false);
        status = gDetector->Init(option);
    } else {
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        status = gDetector->Init(option);
    }

    if (status != TNN_NS::TNN_OK) {
        LOGE("detector init failed %d", (int) status);
        return -1;
    }
    if (clsFaceInfo == NULL) {
        clsFaceInfo = static_cast<jclass>(env->NewGlobalRef(
                env->FindClass("com/tencent/tnn/demo/BlazeFaceDetector$BlazeFaceInfo")));
        midconstructorFaceInfo = env->GetMethodID(clsFaceInfo, "<init>", "()V");
        fidx1 = env->GetFieldID(clsFaceInfo, "x1", "F");
        fidy1 = env->GetFieldID(clsFaceInfo, "y1", "F");
        fidx2 = env->GetFieldID(clsFaceInfo, "x2", "F");
        fidy2 = env->GetFieldID(clsFaceInfo, "y2", "F");
        fidkeypoints = env->GetFieldID(clsFaceInfo, "keypoints", "[[F");
    }
    TNN_NS::BenchOption bench_option;
    bench_option.forward_count = 20;
    gDetector->SetBenchOption(bench_option);
    return 0;
}

void makeBlazefaceDetectOption(std::shared_ptr<TNN_NS::BlazeFaceDetectorOption> &option,
                               std::string &lib_path, std::string &proto_content,
                               std::string &model_content) {
    option->library_path = lib_path;
    option->proto_content = proto_content;
    option->model_content = model_content;
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->input_width = target_width;
    option->input_height = target_height;
//    option->min_score_threshold = 0.75;
    option->min_suppression_threshold = 0.3;
    option->anchor_path = modelPathStr + "/blazeface_anchors.txt";
    LOGE("%s", option->anchor_path.c_str());
}

JNIEXPORT JNICALL jboolean
TNN_BLAZEFACE_DETECTOR(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
    TNN_NS::BlazeFaceDetector tmpDetector;
    std::string protoContent, modelContent, lib_path = "";
    modelPathStr = jstring2string(env, modelPath);
    protoContent = fdLoadFile(modelPathStr + "/blazeface.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/blazeface.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(),
         modelContent.length());

    TNN_NS::Status status = TNN_NS::TNN_OK;
    auto option = std::make_shared<TNN_NS::BlazeFaceDetectorOption>();
    makeBlazefaceDetectOption(option, lib_path, protoContent, modelContent);
    option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;

    tmpDetector.setNpuModelPath(modelPathStr + "/");
    tmpDetector.setCheckNpuSwitch(true);
    TNN_NS::Status ret = tmpDetector.Init(option);
    LOGI("THE ret %s\n", ret.description().c_str());
    return ret == TNN_NS::TNN_OK;
}

JNIEXPORT JNICALL jint TNN_BLAZEFACE_DETECTOR(deinit)(JNIEnv *env, jobject thiz) {
    gDetector = nullptr;
    return 0;
}

JNIEXPORT JNICALL jobjectArray
TNN_BLAZEFACE_DETECTOR(detectFromImage)(JNIEnv *env, jobject thiz, jobject imageSource, jint width,
                                        jint height) {
    AndroidBitmapInfo sourceInfocolor;
    void *sourcePixelscolor;
    int orig_height = height;
    int orig_width = width;

    if (AndroidBitmap_getInfo(env, imageSource, &sourceInfocolor) < 0) {
        return 0;
    }

    if (sourceInfocolor.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return 0;
    }

    if (AndroidBitmap_lockPixels(env, imageSource, &sourcePixelscolor) < 0) {
        return 0;
    }
    //orgin dims
    std::vector<int> origin_dims = {1, 4, orig_height, orig_width};
    std::vector<int> resize_dims = {1, 4, target_height, target_width};
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, origin_dims,
                                                   sourcePixelscolor);

    //here add the resize
    auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, resize_dims);

    TNN_NS::ResizeParam param;
    TNN_NS::MatUtils::Resize(*input_mat, *resize_mat, param, NULL);

    std::shared_ptr<TNN_NS::TNNSDKOutput> output = gDetector->CreateSDKOutput();
    auto status = gDetector->Predict(std::make_shared<TNN_NS::BlazeFaceDetectorInput>(resize_mat),
                                     output);
    if (status != TNN_NS::TNN_OK) {
        return 0;
    }
    AndroidBitmap_unlockPixels(env, imageSource);
    gDetector->ProcessSDKOutput(output);
    std::vector<TNN_NS::BlazeFaceInfo> face_info;
    //check face info list null or not
    if (output && dynamic_cast<TNN_NS::BlazeFaceDetectorOutput *> (output.get())) {
        auto face_output = dynamic_cast<TNN_NS::BlazeFaceDetectorOutput *>(output.get());
        face_info = face_output->face_list;
    } else {
        return 0;
    }
    std::string device = "arm";
    if (gComputeUnitType == 1) {
        device = "gpu";
    } else if (gComputeUnitType == 2) {
        device = "huawei_npu";
    }

    char temp[128] = "";
    sprintf(temp, " device: %s \ntime: ", device.c_str());
    std::string computeUnitTips(temp);
    std::string resultTips = std::string(
            computeUnitTips + gDetector->GetBenchResult().Description());
    setBenchResult(resultTips);

    jobjectArray faceInfoArray;

    if (face_info.size() > 0) {
        faceInfoArray = env->NewObjectArray(face_info.size(), clsFaceInfo, NULL);
        for (int i = 0; i < face_info.size(); i++) {
            jobject objFaceInfo = env->NewObject(clsFaceInfo, midconstructorFaceInfo);
            int keypointsNum = face_info[i].key_points.size();
            auto face_orig = face_info[i].AdjustToViewSize(orig_height, orig_width, 2);
            LOGE("face[%d] %f %f %f %f score %f landmark size %d", i, face_orig.x1, face_orig.y1,
                 face_orig.x2, face_orig.y2, face_orig.score, keypointsNum);
            env->SetFloatField(objFaceInfo, fidx1, face_orig.x1);
            env->SetFloatField(objFaceInfo, fidy1, face_orig.y1);
            env->SetFloatField(objFaceInfo, fidx2, face_orig.x2);
            env->SetFloatField(objFaceInfo, fidy2, face_orig.y2);

            //from here start to create point
            jclass cls1dArr = env->FindClass("[F");
            // Create the returnable jobjectArray with an initial value
            jobjectArray outer = env->NewObjectArray(keypointsNum, cls1dArr, NULL);
            for (int j = 0; j < keypointsNum; j++) {
                jfloatArray inner = env->NewFloatArray(2);
                float temp[] = {face_orig.key_points[j].first, face_orig.key_points[j].second};
                env->SetFloatArrayRegion(inner, 0, 2, temp);
                env->SetObjectArrayElement(outer, j, inner);
                env->DeleteLocalRef(inner);
            }
            env->SetObjectField(objFaceInfo, fidkeypoints, outer);
            env->SetObjectArrayElement(faceInfoArray, i, objFaceInfo);
            env->DeleteLocalRef(objFaceInfo);
        }
        return faceInfoArray;
    }
    return 0;
}

JNIEXPORT JNICALL jobjectArray
TNN_BLAZEFACE_DETECTOR(detectFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width,
                                         jint height, jint rotate) {
    jobjectArray faceInfoArray;
    auto asyncRefDetector = gDetector;
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
    TNN_NS::DimsVector resize_dims = {1, 4, target_height, target_width};

    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, input_dims, rgbaData);
    auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, resize_dims);

    TNN_NS::ResizeParam param;
    TNN_NS::MatUtils::Resize(*input_mat, *resize_mat, param, NULL);

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(resize_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = std::make_shared<TNN_NS::TNNSDKOutput>();
    TNN_NS::Status status = asyncRefDetector->Predict(input, output);

    asyncRefDetector->ProcessSDKOutput(output);
    std::vector<TNN_NS::BlazeFaceInfo> face_info = dynamic_cast<TNN_NS::BlazeFaceDetectorOutput *>(output.get())->face_list;
    LOGE("theithilehtisize %d \n", face_info.size());
    delete[] yuvData;
    delete[] rgbaData;
    if (status != TNN_NS::TNN_OK) {
        LOGE("failed to detect %d", (int) status);
        return 0;
    }

    LOGI("bench result: %s", asyncRefDetector->GetBenchResult().Description().c_str());
    char temp[128] = "";
    std::string device = "arm";
    if (gComputeUnitType == 1) {
        device = "gpu";
    } else if (gComputeUnitType == 2) {
        device = "huawei_npu";
    }
    sprintf(temp, " device: %s \ntime:", device.c_str());
    std::string computeUnitTips(temp);
    std::string resultTips = std::string(
            computeUnitTips + asyncRefDetector->GetBenchResult().Description());
    setBenchResult(resultTips);
    faceInfoArray = env->NewObjectArray(face_info.size(), clsFaceInfo, NULL);

    if (face_info.size() > 0) {
        for (int i = 0; i < face_info.size(); i++) {
            jobject objFaceInfo = env->NewObject(clsFaceInfo, midconstructorFaceInfo);
            int keypointsNum = face_info[i].key_points.size();
            auto face_orig = face_info[i].AdjustToViewSize(width, height, 2);
            LOGE("face[%d] %f %f %f %f score %f landmark size %d", i, face_orig.x1, face_orig.y1,
                 face_orig.x2, face_orig.y2, face_orig.score, keypointsNum);
            env->SetFloatField(objFaceInfo, fidx1, face_orig.x1);
            env->SetFloatField(objFaceInfo, fidy1, face_orig.y1);
            env->SetFloatField(objFaceInfo, fidx2, face_orig.x2);
            env->SetFloatField(objFaceInfo, fidy2, face_orig.y2);

//            //from here start to create point
//            jclass cls1dArr = env->FindClass("[F");
//            // Create the returnable jobjectArray with an initial value
//            jobjectArray outer = env->NewObjectArray(keypointsNum, cls1dArr, NULL);
//            for (int j = 0; j < keypointsNum; j++) {
//                jfloatArray inner = env->NewFloatArray(2);
//                float temp[] = {face_orig.key_points[j].first, face_orig.key_points[j].second};
//                env->SetFloatArrayRegion(inner, 0, 2, temp);
//                env->SetObjectArrayElement(outer, j, inner);
//                env->DeleteLocalRef(inner);
//            }
//            env->SetObjectField(objFaceInfo, fidkeypoints, outer);
            env->SetObjectArrayElement(faceInfoArray, i, objFaceInfo);
            env->DeleteLocalRef(objFaceInfo);
        }
        return faceInfoArray;
    } else {
        return 0;
    }
}