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

#ifndef TNN_EXAMPLES_BASE_TNN_SDK_SAMPLE_H_
#define TNN_EXAMPLES_BASE_TNN_SDK_SAMPLE_H_

#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

#define TNN_SDK_ENABLE_BENCHMARK 1
#define TNN_SDK_USE_NCNN_MODEL 0

#ifndef HAS_OPENCV
#define HAS_OPENCV 0
#endif

namespace TNN_NS {

template<typename T1, typename T2, typename T3>
using triple = std::tuple<T1, T2, T3>;

struct ObjectInfo {
    int image_width = 0;
    int image_height = 0;

    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;

    //key_points <x y>
    std::vector<std::pair<float, float>> key_points = {};
    //key_points_3d <x y z>
    std::vector<triple<float,float,float>> key_points_3d = {};
    //lines connecting key_points
    std::vector<std::pair<int, int>> lines;
    // label
    const char *label = nullptr;
    
    float score = 0;
    int class_id = -1;

    ObjectInfo AdjustToImageSize(int image_height, int image_width);
    /**gravity
     * 0:resize
     * 1:resize fit the view and keep aspect, empty space may be remained zero
     *  2:resize to fill the view and keep aspect, no empty space remain
     */
    ObjectInfo AdjustToViewSize(int view_height, int view_width, int gravity = 2);
    ObjectInfo FlipX();
    ObjectInfo AddOffset(float offset_x, float offset_y);
    float IntersectionRatio(ObjectInfo *obj);
};

struct ImageInfo {
    ImageInfo();
    ImageInfo(std::shared_ptr<Mat>mat);
    ImageInfo(const ImageInfo& info);
    int image_width = 0;
    int image_height = 0;
    int image_channel = 0;
    // 4-channel image data
    std::shared_ptr<char> data;

    ImageInfo FlipX();
};

struct BenchOption {
    int warm_count    = 0;
    int forward_count = 1;
    int create_count  = 1;

    std::string Description();
};

struct BenchResult {
    TNN_NS::Status status;

    // time
    float min   = FLT_MAX;
    float max   = FLT_MIN;
    float avg   = 0;
    float total = 0;
    int count   = 0;

    float diff = 0;

    void Reset();
    int AddTime(float time);
    std::string Description();
};

typedef enum {
    // run on cpu
    TNNComputeUnitsCPU = 0,
    // run on gpu, if failed run on cpu
    TNNComputeUnitsGPU = 1,
    // run on huawei_npu, if failed run on cpu
    TNNComputeUnitsHuaweiNPU = 2,
    // run on openvino
    TNNComputeUnitsOpenvino = 3,
    // run on TensorRT
    TNNComputeUnitsTensorRT = 4,
} TNNComputeUnits;

struct RGBA{
    RGBA(int r = 0, int g = 0, int b = 0, int a = 0) : r(r), g(g), b(b), a(a) {}
    unsigned char r, g, b, a;
};

class TNNSDKUtils {
public:
    static DeviceType GetFallBackDeviceType(DeviceType dev);
};

extern const std::string kTNNSDKDefaultName;
class TNNSDKInput {
public:
    TNNSDKInput(std::shared_ptr<TNN_NS::Mat> mat = nullptr);
    virtual ~TNNSDKInput();

    bool IsEmpty();
    std::shared_ptr<TNN_NS::Mat> GetMat(std::string name = kTNNSDKDefaultName);
    bool AddMat(std::shared_ptr<TNN_NS::Mat> mat, std::string name);

protected:
    std::map<std::string, std::shared_ptr<TNN_NS::Mat> > mat_map_ = {};
};

class TNNSDKOutput : public TNNSDKInput {
public:
    TNNSDKOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~TNNSDKOutput();
};

class TNNSDKOption {
public:
    TNNSDKOption();
    virtual ~TNNSDKOption();

    std::string proto_content = "";
    std::string model_content = "";
    std::string library_path = "";
    TNNComputeUnits compute_units = TNNComputeUnitsCPU;
    Precision precision = PRECISION_AUTO;
    InputShapesMap input_shapes = {};
};

typedef enum {
    TNNInterpNearest = 0,
    TNNInterpLinear  = 1,
    TNNInterpCubic   = 2,
} TNNInterpType;

typedef enum {
    TNNBorderConstant = 0,
    TNNBorderReflect  = 1,
    TNNBorderEdge     = 2,
    TNNBorderReplicate = 3,
    TNNBorderReflect101 = 4,
    TNNBorderWrap = 5,
    
} TNNBorderType;

class TNNSDKSample {
public:
    TNNSDKSample();
    virtual ~TNNSDKSample();
    virtual TNNComputeUnits GetComputeUnits();
    void SetBenchOption(BenchOption option);
    BenchResult GetBenchResult();
    virtual DimsVector GetInputShape(std::string name = kTNNSDKDefaultName);


    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);

    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual MatConvertParam GetConvertParamForOutput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    
    virtual std::shared_ptr<TNN_NS::Mat> ProcessSDKInputMat(std::shared_ptr<TNN_NS::Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);

    void setNpuModelPath(std::string stored_path);
    void setCheckNpuSwitch(bool option);
    
    virtual Status GetCommandQueue(void **command_queue);
    virtual Status DumpBlob(const BlobMap& blob_map, std::string output_dir);
    Status Resize(std::shared_ptr<TNN_NS::Mat> src, std::shared_ptr<TNN_NS::Mat> dst, TNNInterpType interp_type);
    Status Crop(std::shared_ptr<TNN_NS::Mat> src, std::shared_ptr<TNN_NS::Mat> dst, int start_x, int start_y);
    Status WarpAffine(std::shared_ptr<TNN_NS::Mat> src, std::shared_ptr<TNN_NS::Mat> dst, TNNInterpType interp_type, TNNBorderType border_type, float trans_mat[2][3]);
    Status Copy(std::shared_ptr<TNN_NS::Mat> src, std::shared_ptr<TNN_NS::Mat> dst);
    Status CopyMakeBorder(std::shared_ptr<TNN_NS::Mat> src,
                          std::shared_ptr<TNN_NS::Mat> dst,
                          int top, int bottom, int left, int right,
                          TNNBorderType border_type, uint8_t border_value = 0);
    virtual bool hideTextBox();

protected:
    BenchOption bench_option_;
    BenchResult bench_result_;

    std::vector<std::string> GetInputNames();
    std::vector<std::string> GetOutputNames();
    std::shared_ptr<Mat> ResizeToInputShape(std::shared_ptr<Mat> input_mat, std::string name);
    
protected:
    std::shared_ptr<TNN> net_             = nullptr;
    std::shared_ptr<Instance> instance_   = nullptr;
    std::shared_ptr<TNNSDKOption> option_ = nullptr;
    DeviceType device_type_               = DEVICE_ARM;
    std::string model_path_str_           = "";
    bool check_npu_                       = false;
};

class TNNSDKComposeSample : public TNNSDKSample {
public:
    TNNSDKComposeSample();
    virtual ~TNNSDKComposeSample();
    virtual TNNComputeUnits GetComputeUnits();
    
    virtual Status Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks);
    virtual DimsVector GetInputShape(std::string name = kTNNSDKDefaultName);
    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);
    virtual Status GetCommandQueue(void **command_queue);
    
protected:
    std::vector<std::shared_ptr<TNNSDKSample>> sdks_ = {};
    
};

typedef enum {
    TNNHardNMS      = 0,
    TNNBlendingNMS  = 1,
    TNNWeightedNMS  = 2,
} TNNNMSType;

void NMS(std::vector<ObjectInfo> &input, std::vector<ObjectInfo> &output, float iou_threshold, TNNNMSType type);

void Rectangle(void *data_rgba, int image_height, int image_width,
               int x0, int y0, int x1, int y1, float scale_x = 1.0, float scale_y = 1.0);

void Point(void *data_rgba, int image_height, int image_width,
           int x, int y, float z, float scale_x = 1.0, float scale_y = 1.0);
}  // namespace TNN_NS

#endif // TNN_EXAMPLES_BASE_TNN_SDK_SAMPLE_H_
