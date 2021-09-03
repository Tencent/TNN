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

#include "tnn_sdk_sample.h"
#include "sample_timer.h"
#include "tnn/utils/dims_vector_utils.h"
#include <algorithm>
#include <cstring>
#include <float.h>

#if defined(__APPLE__)
#include "TargetConditionals.h"
#endif

#define ENABLE_DUMP_BLOB_DATA 0
#if ENABLE_DUMP_BLOB_DATA
static int blob_id = 0;
#endif

namespace TNN_NS {
const std::string kTNNSDKDefaultName = "TNN.sdk.default.name";

void printShape(const std::string& msg, const DimsVector& shape) {
    printf("%s:(%d,%d,%d,%d)\n", msg.c_str(), shape[0], shape[1], shape[2], shape[3]);
}
ImageInfo::ImageInfo() {
    image_width = 0;
    image_height = 0;
    image_channel = 0;
    data = nullptr;
}

ImageInfo::ImageInfo(const ImageInfo& info) {
    image_width = info.image_width;
    image_height = info.image_height;
    image_channel = info.image_channel;
    data = info.data;
}

ImageInfo::ImageInfo(std::shared_ptr<Mat>image) {
    if (image != nullptr) {
        const auto& dims = image->GetDims();
        image_channel = dims[1];
        image_height = dims[2];
        image_width = dims[3];
        auto count = DimsVectorUtils::Count(dims);
        data.reset(new char[count]);
        memcpy(data.get(), image->GetData(), count);
    }
}

ImageInfo ImageInfo::FlipX() {
    auto flip_image_row = [](const char*src, char*dst, int width, int channel){
        int src_offset = (width-1) * channel;
        int dst_offset = 0;
        for(int w=0; w<width; ++w) {
            for(int c=0; c<channel; ++c) {
                dst[dst_offset + c] = src[src_offset + c];
            }
            src_offset -= channel;
            dst_offset += channel;
        }
    };
    ImageInfo info;
    info.image_width = image_width;
    info.image_height = image_height;
    info.image_channel = image_channel;
    info.data.reset(new char[info.image_height * info.image_width*info.image_channel]);
    auto bytes_per_row = image_width * image_channel;
    for(int h=0; h<image_height; ++h) {
        flip_image_row(data.get()+h*bytes_per_row, info.data.get()+h*bytes_per_row, image_width, image_channel);
    }

    return info;
}

ObjectInfo ObjectInfo::FlipX() {
    ObjectInfo  info;
    info.score = this->score;
    info.class_id = this->class_id;
    info.image_width = this->image_width;
    info.image_height = this->image_width;
    info.lines = this->lines;
    
    info.x1 = this->image_width - this->x2;
    info.x2 = this->image_width - this->x1;
    info.y1 = this->y1;
    info.y2 = this->y2;
    
    //key points
    std::vector<std::pair<float, float>> key_points;
    for (auto item : this->key_points) {
        key_points.push_back(std::make_pair(this->image_width - item.first, item.second));
    }
    info.key_points = key_points;
    
    //key points 3d
    std::vector<triple<float, float, float>> key_points_3d;
    for (auto item : this->key_points_3d) {
        key_points_3d.push_back(std::make_tuple(this->image_width - std::get<0>(item),
                                                                     std::get<1>(item),
                                                                     std::get<2>(item)));
    }
    info.key_points_3d = key_points_3d;
    return info;
}

ObjectInfo ObjectInfo::AddOffset(float offset_x, float offset_y) {
    ObjectInfo  info;
    info.score = this->score;
    info.class_id = this->class_id;
    info.image_width = this->image_width;
    info.image_height = this->image_width;
    
    info.x1 = this->x1 + offset_x;
    info.x2 = this->x2 + offset_x;
    info.y1 = this->y1 + offset_y;
    info.y2 = this->y2 + offset_y;
    
    //key points
    std::vector<std::pair<float, float>> key_points;
    for (auto item : this->key_points) {
        key_points.push_back(std::make_pair(item.first + offset_x, item.second + offset_y));
    }
    info.key_points = key_points;
    
    //key points 3d
    std::vector<triple<float, float, float>> key_points_3d;
    for (auto item : this->key_points_3d) {
        key_points_3d.push_back(std::make_tuple(std::get<0>(item) + offset_x,
                                                                     std::get<1>(item) + offset_y,
                                                                     std::get<2>(item)));
    }
    info.key_points_3d = key_points_3d;
    return info;
}

float ObjectInfo::IntersectionRatio(ObjectInfo *obj) {
    if (!obj) {
        return 0;
    }
    
    float area1 = std::abs((this->x2 - this->x1) * (this->y2 - this->y1));
    float area2 = std::abs((obj->x2 - obj->x1) * (obj->y2 - obj->y1));
    
    float x1 = std::max(obj->x1, this->x1);
    float x2 = std::min(obj->x2, this->x2);
    float y1 = std::max(obj->y1, this->y1);
    float y2 = std::min(obj->y2, this->y2);
    
    float area = (x2 > x1 && y2 > y1) ? std::abs((x2 - x1) * (y2 - y1)) : 0;
    
    return area / (area1 + area2 - area);
}

ObjectInfo ObjectInfo::AdjustToImageSize(int orig_image_height, int orig_image_width) {
    float scale_x = orig_image_width/(float)this->image_width;
    float scale_y = orig_image_height/(float)this->image_height;
    
    ObjectInfo  info_orig;
    info_orig.score = this->score;
    info_orig.class_id = this->class_id;
    info_orig.image_width = orig_image_width;
    info_orig.image_height = orig_image_height;
    
    int x_min = std::min(this->x1, this->x2)*scale_x;
    int x_max = std::max(this->x1, this->x2)*scale_x;
    int y_min = std::min(this->y1, this->y2)*scale_y;
    int y_max = std::max(this->y1, this->y2)*scale_y;
    
    x_min = std::min(std::max(x_min, 0), orig_image_width-1);
    x_max = std::min(std::max(x_max, 0), orig_image_width-1);
    y_min = std::min(std::max(y_min, 0), orig_image_height-1);
    y_max = std::min(std::max(y_max, 0), orig_image_height-1);
    
    info_orig.x1 = x_min;
    info_orig.x2 = x_max;
    info_orig.y1 = y_min;
    info_orig.y2 = y_max;
    
    
    //key points
    std::vector<std::pair<float, float>> key_points;
    for (auto item : this->key_points) {
        key_points.push_back(std::make_pair(item.first*scale_x, item.second*scale_y));
    }
    info_orig.key_points = key_points;
    
    //key points 3d
    std::vector<triple<float, float, float>> key_points_3d;
    for (auto item : this->key_points_3d) {
        key_points_3d.push_back(std::make_tuple(std::get<0>(item) * scale_x,
                                                                     std::get<1>(item) * scale_y,
                                                                     std::get<2>(item)));
    }
    info_orig.key_points_3d = key_points_3d;
    info_orig.lines = lines;
    
    return info_orig;
}

ObjectInfo ObjectInfo::AdjustToViewSize(int view_height, int view_width, int gravity) {
    ObjectInfo  info;
    info.score = this->score;
    info.class_id = this->class_id;
    info.image_width = view_width;
    info.image_height = view_height;
    info.lines = lines;
    
    float view_aspect = view_height/(float)(view_width + FLT_EPSILON);
    float object_aspect = this->image_height/(float)(this->image_width + FLT_EPSILON);
    
    if (gravity == 2) {
        if (view_aspect > object_aspect) {
            float object_aspect_width = view_height / object_aspect;
            auto info_aspect = AdjustToImageSize(view_height, object_aspect_width);
            float offset_x = (object_aspect_width - view_width) / 2;
            info_aspect = info_aspect.AddOffset(-offset_x, 0);
            info.x1 = info_aspect.x1;
            info.x2 = info_aspect.x2;
            info.y1 = info_aspect.y1;
            info.y2 = info_aspect.y2;
            info.key_points = info_aspect.key_points;
            info.key_points_3d = info_aspect.key_points_3d;
        } else {
            float object_aspect_height = view_width * object_aspect;
            auto info_aspect = AdjustToImageSize(object_aspect_height, view_width);
            float offset_y = (object_aspect_height - view_height) / 2;
            info_aspect = info_aspect.AddOffset(0, -offset_y);
            info.x1 = info_aspect.x1;
            info.x2 = info_aspect.x2;
            info.y1 = info_aspect.y1;
            info.y2 = info_aspect.y2;
            info.key_points = info_aspect.key_points;
            info.key_points_3d = info_aspect.key_points_3d;
        }
    } else if (gravity == 1) {
        if (view_aspect > object_aspect) {
            float object_aspect_height = view_width * object_aspect;
            auto info_aspect = AdjustToImageSize(object_aspect_height, view_width);
            float offset_y = (object_aspect_height - view_height) / 2;
            info_aspect = info_aspect.AddOffset(0, -offset_y);
            info.x1 = info_aspect.x1;
            info.x2 = info_aspect.x2;
            info.y1 = info_aspect.y1;
            info.y2 = info_aspect.y2;
            info.key_points = info_aspect.key_points;
            info.key_points_3d = info_aspect.key_points_3d;
        } else {
            float object_aspect_width = view_height / object_aspect;
            auto info_aspect = AdjustToImageSize(view_height, object_aspect_width);
            float offset_x = (object_aspect_width - view_width) / 2;
            info_aspect = info_aspect.AddOffset(-offset_x, 0);
            info.x1 = info_aspect.x1;
            info.x2 = info_aspect.x2;
            info.y1 = info_aspect.y1;
            info.y2 = info_aspect.y2;
            info.key_points = info_aspect.key_points;
            info.key_points_3d = info_aspect.key_points_3d;
        }
    } else {
        return AdjustToImageSize(view_height, view_width);
    }
    return info;
}

std::string BenchOption::Description() {
    std::ostringstream ostr;
    ostr << "create_count = " << create_count << "  warm_count = " << warm_count
         << "  forward_count = " << forward_count;

    ostr << std::endl;
    return ostr.str();
}

void BenchResult::Reset() {
    min   = FLT_MAX;
    max   = FLT_MIN;
    avg   = 0;
    total = 0;
    count = 0;

    diff = 0;
}

int BenchResult::AddTime(float time) {
    count++;
    total += time;
    min = std::min(min, time);
    max = std::max(max, time);
    avg = total / count;
    return 0;
}

std::string BenchResult::Description() {
    std::ostringstream ostr;
    ostr << "min = " << min << "  max = " << max << "  avg = " << avg;

    if (status != TNN_NS::TNN_OK) {
        ostr << "\nerror = " << status.description();
    }
    ostr << std::endl;

    return ostr.str();
}

DeviceType TNNSDKUtils::GetFallBackDeviceType(DeviceType dev) {
    switch (dev) {
        case DEVICE_CUDA:
            return DEVICE_NAIVE;
        case DEVICE_RK_NPU:
        case DEVICE_HUAWEI_NPU:
        case DEVICE_METAL:
        case DEVICE_OPENCL:
        case DEVICE_ATLAS:
        case DEVICE_DSP:
            return DEVICE_ARM;
        case DEVICE_X86:
        case DEVICE_ARM:
        case DEVICE_NAIVE:
            return dev;
    }
    return DEVICE_NAIVE;
}

#pragma mark - TNNSDKInput
TNNSDKInput::TNNSDKInput(std::shared_ptr<TNN_NS::Mat> mat) {
    if (mat) {
        mat_map_[kTNNSDKDefaultName] = mat;
    }
}

TNNSDKInput::~TNNSDKInput() {}

bool TNNSDKInput::IsEmpty() {
    if (mat_map_.size() <= 0) {
        return true;
    }
    return false;
}

bool TNNSDKInput::AddMat(std::shared_ptr<TNN_NS::Mat> mat, std::string name) {
    if (name.empty() || !mat) {
        return false;
    }
    
    mat_map_[name] = mat;
    return true;
}

std::shared_ptr<TNN_NS::Mat> TNNSDKInput::GetMat(std::string name) {
    std::shared_ptr<TNN_NS::Mat> mat = nullptr;
    if (name == kTNNSDKDefaultName && mat_map_.size() > 0) {
        return mat_map_.begin()->second;
    }
    
    if (mat_map_.find(name) != mat_map_.end()) {
        mat = mat_map_[name];
    }
    return mat;
}

#pragma mark - TNNSDKOutput
TNNSDKOutput::~TNNSDKOutput() {}

#pragma mark - TNNSDKOption
TNNSDKOption::TNNSDKOption() {}

TNNSDKOption::~TNNSDKOption() {}

#pragma mark - TNNSDKSample
TNNSDKSample::TNNSDKSample() {}

TNNSDKSample::~TNNSDKSample() {}


void TNNSDKSample::setCheckNpuSwitch(bool option)
{
    check_npu_ = option;
}

Status TNNSDKSample::GetCommandQueue(void **command_queue) {
    if (instance_) {
        return instance_->GetCommandQueue(command_queue);
    }
    return Status(TNNERR_INST_ERR, "instance_ GetCommandQueue return nil");
}

Status TNNSDKSample::Resize(std::shared_ptr<TNN_NS::Mat> src, std::shared_ptr<TNN_NS::Mat> dst, TNNInterpType interp_type) {
    Status status = TNN_OK;
    
    void * command_queue = nullptr;
    status = GetCommandQueue(&command_queue);
    if (status != TNN_NS::TNN_OK) {
        LOGE("getCommandQueue failed with:%s\n", status.description().c_str());
        return status;
    }
    
    InterpType type = INTERP_TYPE_NEAREST;
    if(interp_type == TNNInterpNearest){
        type = TNN_NS::INTERP_TYPE_NEAREST;
    } else if(interp_type == TNNInterpLinear) {
        type = TNN_NS::INTERP_TYPE_LINEAR;
    }
    
    ResizeParam param;
    param.type = type;
    
    auto dst_dims = dst->GetDims();
    auto src_dims = src->GetDims();
    param.scale_w = dst_dims[3] / static_cast<float>(src_dims[3]);
    param.scale_h = dst_dims[2] / static_cast<float>(src_dims[2]);
    
    status = MatUtils::Resize(*(src.get()), *(dst.get()), param, command_queue);
    if (status != TNN_NS::TNN_OK){
        LOGE("resize failed with:%s\n", status.description().c_str());
    }
    
    return status;
}

Status TNNSDKSample::Crop(std::shared_ptr<TNN_NS::Mat> src, std::shared_ptr<TNN_NS::Mat> dst, int start_x, int start_y) {
    Status status = TNN_OK;
    
    void *command_queue = nullptr;
    status = GetCommandQueue(&command_queue);
    if (status != TNN_NS::TNN_OK) {
        LOGE("getCommandQueue failed with:%s\n", status.description().c_str());
        return status;
    }
    
    CropParam param;
    param.top_left_x = start_x;
    param.top_left_y = start_y;
    auto dst_dims = dst->GetDims();
    param.width  = dst_dims[3];
    param.height = dst_dims[2];
    
    status = MatUtils::Crop(*(src.get()), *(dst.get()), param, command_queue);
    if (status != TNN_NS::TNN_OK){
        LOGE("crop failed with:%s\n", status.description().c_str());
    }
    
    return status;
}

Status TNNSDKSample::WarpAffine(std::shared_ptr<TNN_NS::Mat> src, std::shared_ptr<TNN_NS::Mat> dst, TNNInterpType interp_type, TNNBorderType border_type, float trans_mat[2][3]) {
    Status status = TNN_OK;
    
    void * command_queue = nullptr;
    status = GetCommandQueue(&command_queue);
    if (status != TNN_OK) {
        LOGE("getCommandQueue failed with:%s\n", status.description().c_str());
        return status;
    }
    
    InterpType itype = INTERP_TYPE_NEAREST;
    if (interp_type == TNNInterpNearest){
        itype = INTERP_TYPE_NEAREST;
    } else if(interp_type == TNNInterpLinear) {
        itype = INTERP_TYPE_LINEAR;
    }
    BorderType btype = BORDER_TYPE_CONSTANT;
    if (border_type == TNNBorderConstant) {
        btype = BORDER_TYPE_CONSTANT;
    } else if(border_type == TNNBorderReflect) {
        btype = BORDER_TYPE_REFLECT;
    } else if(border_type == TNNBorderEdge) {
        btype = BORDER_TYPE_EDGE;
    }
    WarpAffineParam param;
    param.interp_type = itype;
    param.border_type = btype;
    
    auto dst_dims = dst->GetDims();
    auto src_dims = src->GetDims();
    memcpy(param.transform, trans_mat, sizeof(float)*2*3);
    
    status = MatUtils::WarpAffine(*(src.get()), *(dst.get()), param, command_queue);
    if (status != TNN_NS::TNN_OK){
        LOGE("warpaffine failed with:%s\n", status.description().c_str());
    }
    
    return status;
}

Status TNNSDKSample::Copy(std::shared_ptr<TNN_NS::Mat> src, std::shared_ptr<TNN_NS::Mat> dst) {
    Status status = TNN_OK;
    
    void *command_queue = nullptr;
    status = GetCommandQueue(&command_queue);
    if (status != TNN_NS::TNN_OK) {
        LOGE("getCommandQueue failed with:%s\n", status.description().c_str());
        return status;
    }
    
    status = MatUtils::Copy(*(src.get()), *(dst.get()), command_queue);
    if (status != TNN_NS::TNN_OK){
        LOGE("copy failed with:%s\n", status.description().c_str());
    }
    
    return status;
}

Status TNNSDKSample::CopyMakeBorder(std::shared_ptr<TNN_NS::Mat> src,
                      std::shared_ptr<TNN_NS::Mat> dst,
                      int top, int bottom, int left, int right,
                                    TNNBorderType border_type, uint8_t border_value) {
    Status status = TNN_OK;
    
    void *command_queue = nullptr;
    status = GetCommandQueue(&command_queue);
    if (status != TNN_NS::TNN_OK) {
        LOGE("getCommandQueue failed with:%s\n", status.description().c_str());
        return status;
    }
    
    CopyMakeBorderParam param;
    param.border_val = border_value;
    param.top = top;
    param.bottom = bottom;
    param.left = left;
    param.right = right;
    param.border_type = BORDER_TYPE_CONSTANT;
    if (border_type == TNNBorderEdge)
        param.border_type = BORDER_TYPE_EDGE;
    else if (border_type == TNNBorderReflect)
        param.border_type = BORDER_TYPE_REFLECT;
    
    status = MatUtils::CopyMakeBorder(*(src.get()), *(dst.get()), param, command_queue);
    if (status != TNN_NS::TNN_OK){
        LOGE("copy failed with:%s\n", status.description().c_str());
    }
    
    return status;
}

void TNNSDKSample::setNpuModelPath(std::string stored_path)
{
    model_path_str_ = stored_path;
}

TNN_NS::Status TNNSDKSample::Init(std::shared_ptr<TNNSDKOption> option) {
    option_ = option;
    //网络初始化
    TNN_NS::Status status;
    if (!net_) {
        TNN_NS::ModelConfig config;
#if TNN_SDK_USE_NCNN_MODEL
        config.model_type = TNN_NS::MODEL_TYPE_NCNN;
#else
        config.model_type = TNN_NS::MODEL_TYPE_TNN;
#endif
        config.params = {option->proto_content, option->model_content, model_path_str_};

        auto net = std::make_shared<TNN_NS::TNN>();
        status   = net->Init(config);
        if (status != TNN_NS::TNN_OK) {
            LOGE("instance.net init failed %d", (int)status);
            return status;
        }
        net_ = net;
    }

    // network init
#if defined(TNN_USE_NEON)
    device_type_ = TNN_NS::DEVICE_ARM;
#else
    device_type_ = TNN_NS::DEVICE_X86;
#endif
    if(option->compute_units == TNNComputeUnitsGPU) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
        device_type_ = TNN_NS::DEVICE_METAL;
#else
        device_type_ = TNN_NS::DEVICE_OPENCL;
#endif
    }
    else if (option->compute_units == TNNComputeUnitsHuaweiNPU) {
        device_type_      = TNN_NS::DEVICE_HUAWEI_NPU;
#if defined(__APPLE__) && TARGET_OS_IPHONE
        device_type_ = TNN_NS::DEVICE_METAL;
#else
        device_type_      = TNN_NS::DEVICE_HUAWEI_NPU;
#endif
    } else if (option->compute_units == TNNComputeUnitsTensorRT) {
        device_type_ = TNN_NS::DEVICE_CUDA;
    }
    
    //创建实例instance
    {
        TNN_NS::NetworkConfig network_config;
        network_config.library_path = {option->library_path};
        network_config.device_type  = device_type_;
        network_config.precision = option->precision;
        network_config.cache_path = "/sdcard/";
        if(device_type_ == TNN_NS::DEVICE_HUAWEI_NPU){
            network_config.network_type = NETWORK_TYPE_HUAWEI_NPU;
        } else if (option->compute_units == TNNComputeUnitsOpenvino) {
            network_config.network_type = NETWORK_TYPE_OPENVINO;
        } else if (device_type_ == TNN_NS::DEVICE_CUDA) {
            network_config.network_type = NETWORK_TYPE_TENSORRT;
        }
        auto instance               = net_->CreateInst(network_config, status, option->input_shapes);

        if (!check_npu_ && (status != TNN_NS::TNN_OK || !instance)) {
            // try device_arm
            if (option->compute_units >= TNNComputeUnitsGPU) {
                device_type_               = TNN_NS::DEVICE_ARM;
                network_config.device_type = TNN_NS::DEVICE_ARM;
                instance                   = net_->CreateInst(network_config, status,  option->input_shapes);
            }
        }
        instance_ = instance;
    }
    return status;
}

TNNComputeUnits TNNSDKSample::GetComputeUnits() {
    switch (device_type_) {
        case DEVICE_HUAWEI_NPU:
            return TNNComputeUnitsHuaweiNPU;
        case DEVICE_METAL:
        case DEVICE_OPENCL:
            return TNNComputeUnitsGPU;
        default:
            return TNNComputeUnitsCPU;
    }
}

void TNNSDKSample::SetBenchOption(BenchOption option) {
    bench_option_ = option;
}

BenchResult TNNSDKSample::GetBenchResult() {
    return bench_result_;
}

DimsVector TNNSDKSample::GetInputShape(std::string name) {
    DimsVector shape = {};
    BlobMap blob_map = {};
    if (instance_) {
        instance_->GetAllInputBlobs(blob_map);
    }
    
    if (kTNNSDKDefaultName == name && blob_map.size() > 0) {
        if (blob_map.begin()->second) {
            shape = blob_map.begin()->second->GetBlobDesc().dims;
        }
    }
    
    if (blob_map.find(name) != blob_map.end() && blob_map[name]) {
        shape = blob_map[name]->GetBlobDesc().dims;
    }
    return shape;
}

std::vector<std::string> TNNSDKSample::GetInputNames() {
    std::vector<std::string> names;
    if (instance_) {
        BlobMap blob_map;
        instance_->GetAllInputBlobs(blob_map);
        for (const auto& item : blob_map) {
            names.push_back(item.first);
        }
    }
    return names;
}

std::vector<std::string> TNNSDKSample::GetOutputNames() {
    std::vector<std::string> names;
    if (instance_) {
        BlobMap blob_map;
        instance_->GetAllOutputBlobs(blob_map);
        for (const auto& item : blob_map) {
            names.push_back(item.first);
        }
    }
    return names;
}

std::shared_ptr<Mat> TNNSDKSample::ResizeToInputShape(std::shared_ptr<Mat> input_mat, std::string name) {
    auto target_dims = GetInputShape(name);
    auto input_height = input_mat->GetHeight();
    auto input_width = input_mat->GetWidth();
    if (target_dims.size() >= 4 &&
        (input_height != target_dims[2] || input_width != target_dims[3])) {
        auto target_mat = std::make_shared<TNN_NS::Mat>(input_mat->GetDeviceType(),
                                                        input_mat->GetMatType(), target_dims);
        auto status = Resize(input_mat, target_mat, TNNInterpLinear);
        if (status == TNN_OK) {
            return target_mat;
        } else {
            LOGE("%s\n", status.description().c_str());
            return nullptr;
        }
    }
    return input_mat;
}

bool TNNSDKSample::hideTextBox() {
    return false;
}

TNN_NS::MatConvertParam TNNSDKSample::GetConvertParamForInput(std::string name) {
    return TNN_NS::MatConvertParam();
}

TNN_NS::MatConvertParam TNNSDKSample::GetConvertParamForOutput(std::string name) {
    return TNN_NS::MatConvertParam();
}

std::shared_ptr<TNNSDKOutput> TNNSDKSample::CreateSDKOutput() {
    return std::make_shared<TNNSDKOutput>();
}

TNN_NS::Status TNNSDKSample::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output) {
    return TNN_OK;
}

std::shared_ptr<TNN_NS::Mat> TNNSDKSample::ProcessSDKInputMat(std::shared_ptr<TNN_NS::Mat> mat,
                                                              std::string name) {
    return mat;
}

TNN_NS::Status TNNSDKSample::DumpBlob(const BlobMap& blob_map, std::string output_dir) {
#if ENABLE_DUMP_BLOB_DATA
    for (const auto& item : blob_map) {
        std::string output_path = output_dir + "/" + item.first + "_" + std::to_string(blob_id++);
        DeviceType device_type = DEVICE_NAIVE;
        MatType mat_type = NCHW_FLOAT;

        void* command_queue;
        instance_->GetCommandQueue(&command_queue);
        BlobConverter blob_converter(item.second);
        MatConvertParam param;
        Mat cpu_mat(device_type, mat_type, item.second->GetBlobDesc().dims);
        Status ret = blob_converter.ConvertToMat(cpu_mat, param, command_queue);
        if (ret != TNN_OK) {
            LOGE("blob (name: %s) convert failed (%s)\n", item.first.c_str(), ret.description().c_str());
            return ret;
        }

        std::ofstream out_stream(output_path);
        if (out_stream.is_open()) {
            out_stream << item.first << std::endl;
            for (auto d : cpu_mat.GetDims()) {
                out_stream << d << " ";
            }
            out_stream << std::endl;
            float* data_ptr = reinterpret_cast<float*>(cpu_mat.GetData());
            for (int index = 0; index < DimsVectorUtils::Count(cpu_mat.GetDims()); index++) {
                out_stream << data_ptr[index] << std::endl;
            }
            out_stream.close();
        }
    }
#endif
    return TNN_OK;
}

TNN_NS::Status TNNSDKSample::Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output) {
    Status status = TNN_OK;
    if (!input || input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty ,please check!\n");
        return status;
    }
    
#if TNN_SDK_ENABLE_BENCHMARK
    bench_result_.Reset();
    for (int fcount = 0; fcount < bench_option_.forward_count; fcount++) {
        SampleTimer sample_time;
        sample_time.Start();
#endif
        
        // step 1. set input mat
        auto input_names = GetInputNames();
        if (input_names.size() == 1) {
            auto input_mat = input->GetMat();
            input_mat = ProcessSDKInputMat(input_mat);
            auto input_convert_param = GetConvertParamForInput();
            auto status = instance_->SetInputMat(input_mat, input_convert_param);
            RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
        } else {
            for (auto name : input_names) {
                auto input_mat = input->GetMat(name);
                input_mat = ProcessSDKInputMat(input_mat, name);
                auto input_convert_param = GetConvertParamForInput(name);
                auto status = instance_->SetInputMat(input_mat, input_convert_param, name);
                RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
            }
        }

#if ENABLE_DUMP_BLOB_DATA
        BlobMap blob_map = {};
        instance_->GetAllInputBlobs(blob_map);
        std::string output_dir = "/mnt/sdcard";
        status = DumpBlob(blob_map, output_dir);
        if (status != TNN_NS::TNN_OK) {
            LOGE("Dump Blob Error: %s\n", status.description().c_str());
            return status;
        }
#endif
        
        // step 2. Forward
        status = instance_->ForwardAsync(nullptr);
        if (status != TNN_NS::TNN_OK) {
            LOGE("instance.Forward Error: %s\n", status.description().c_str());
            return status;
        }

        // step 3. get output mat
        auto input_device_type = input->GetMat()->GetDeviceType();
        output = CreateSDKOutput();
        auto output_names = GetOutputNames();
        if (output_names.size() == 1) {
            auto output_convert_param = GetConvertParamForOutput();
            std::shared_ptr<TNN_NS::Mat> output_mat = nullptr;
            status = instance_->GetOutputMat(output_mat, output_convert_param, "",
                                             TNNSDKUtils::GetFallBackDeviceType(input_device_type));
            RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
            output->AddMat(output_mat, output_names[0]);
        } else {
            for (auto name : output_names) {
                auto output_convert_param = GetConvertParamForOutput(name);
                std::shared_ptr<TNN_NS::Mat> output_mat = nullptr;
                status = instance_->GetOutputMat(output_mat, output_convert_param, name,
                                                 TNNSDKUtils::GetFallBackDeviceType(input_device_type));
                RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
                output->AddMat(output_mat, name);
            }
        }
  
        
#if TNN_SDK_ENABLE_BENCHMARK
        sample_time.Stop();
        double elapsed = sample_time.GetTime();
        bench_result_.AddTime(elapsed);
#endif
        
        ProcessSDKOutput(output);
#if TNN_SDK_ENABLE_BENCHMARK
    }
#endif
    // Detection done
    
    return status;
}

#pragma mark - TNNSDKComposeSample
TNNSDKComposeSample::TNNSDKComposeSample() {}

TNNSDKComposeSample::~TNNSDKComposeSample() {
    sdks_ = {};
}

Status TNNSDKComposeSample::Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks) {
    sdks_ = sdks;
    return TNN_OK;
}

TNNComputeUnits TNNSDKComposeSample::GetComputeUnits() {
    if (sdks_.size() > 0) {
        return sdks_[0]->GetComputeUnits();
    }
    return TNNComputeUnitsCPU;
}

Status TNNSDKComposeSample::GetCommandQueue(void **command_queue) {
    if (sdks_.size() > 0) {
        return sdks_[0]->GetCommandQueue(command_queue);
    }
    return Status(TNNERR_INST_ERR, "instance_ GetCommandQueue return nil");
}

DimsVector TNNSDKComposeSample::GetInputShape(std::string name) {
    DimsVector shape = {};
    if (sdks_.size() > 0) {
        return sdks_[0]->GetInputShape(name);
    }
    return shape;
}

TNN_NS::Status TNNSDKComposeSample::Predict(std::shared_ptr<TNNSDKInput> input,
                                            std::shared_ptr<TNNSDKOutput> &output) {
    LOGE("subclass of TNNSDKComposeSample must implement this interface\n");
    return Status(TNNERR_NO_RESULT, "subclass of TNNSDKComposeSample must implement this interface");
}

/*
* NMS, supporting hard-nms, blending-nms and weighted-nms
*/
void NMS(std::vector<ObjectInfo> &input, std::vector<ObjectInfo> &output, float iou_threshold, TNNNMSType type) {
    std::sort(input.begin(), input.end(), [](const ObjectInfo &a, const ObjectInfo &b) { return a.score > b.score; });
    output.clear();

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<ObjectInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case TNNHardNMS: {
                output.push_back(buf[0]);
                break;
            }
            case TNNBlendingNMS: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].score);
                }
                ObjectInfo rects;
                memset(&rects, 0, sizeof(rects));
                rects.key_points.resize(buf[0].key_points.size());
                for (int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                    for(int j = 0; j < buf[i].key_points.size(); ++j) {
                        rects.key_points[j].first += buf[i].key_points[j].first * rate;
                        rects.key_points[j].second += buf[i].key_points[j].second * rate;
                    }
                    rects.image_height = buf[0].image_height;
                    rects.image_width  = buf[0].image_width;
                }
                output.push_back(rects);
                break;
            }
            case TNNWeightedNMS: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += buf[i].score;
                }
                ObjectInfo rects;
                memset(&rects, 0, sizeof(rects));
                rects.key_points.resize(buf[0].key_points.size());
                for (int i = 0; i < buf.size(); i++) {
                    float rate = buf[i].score / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                    for(int j = 0; j < buf[i].key_points.size(); ++j) {
                        rects.key_points[j].first += buf[i].key_points[j].first * rate;
                        rects.key_points[j].second += buf[i].key_points[j].second * rate;
                    }
                    rects.image_height = buf[0].image_height;
                    rects.image_width  = buf[0].image_width;
                }
                output.push_back(rects);
                break;
            }
            default: {
            }
        }
    }
}

/*
 * Rectangle
 */
void Rectangle(void *data_rgba, int image_height, int image_width,
               int x0, int y0, int x1, int y1, float scale_x, float scale_y)
{

    
    RGBA *image_rgba = (RGBA *)data_rgba;

    int x_min = std::min(x0, x1) * scale_x;
    int x_max = std::max(x0, x1) * scale_x;
    int y_min = std::min(y0, y1) * scale_y;
    int y_max = std::max(y0, y1) * scale_y;

    x_min = std::min(std::max(x_min, 0), image_width - 1);
    x_max = std::min(std::max(x_max, 0), image_width - 1);
    y_min = std::min(std::max(y_min, 0), image_height - 1);
    y_max = std::min(std::max(y_max, 0), image_height - 1);

    // top bottom
    if (x_max > x_min) {
        for (int x = x_min; x <= x_max; x++) {
            int offset                       = y_min * image_width + x;
            image_rgba[offset]               = {0, 255, 0, 0};
            image_rgba[offset + image_width] = {0, 255, 0, 0};

            offset                           = y_max * image_width + x;
            image_rgba[offset]               = {0, 255, 0, 0};
            if (offset >= image_width) {
                image_rgba[offset - image_width] = {0, 255, 0, 0};
            }
        }
    }

    // left right
    if (y_max > y_min) {
        for (int y = y_min; y <= y_max; y++) {
            int offset             = y * image_width + x_min;
            image_rgba[offset]     = {0, 255, 0, 0};
            image_rgba[offset + 1] = {0, 255, 0, 0};

            offset                 = y * image_width + x_max;
            image_rgba[offset]     = {0, 255, 0, 0};
            if (offset >= 1) {
                image_rgba[offset - 1] = {0, 255, 0, 0};
            }
        }
    }
}

/*
 * Point
 */
void Point(void *data_rgba, int image_height, int image_width, int x, int y, float z, float scale_x, float scale_y)
{
    RGBA *image_rgba = (RGBA *)data_rgba;
    int x_center = x * scale_x;
    int y_center = y * scale_y;
    int x_start = (x-1) * scale_x;
    int x_end   = (x+1) * scale_x;
    int y_start = (y-1) * scale_y;
    int y_end   = (y+1) * scale_y;

    x_center = std::min(std::max(0, x_center), image_width  - 1);
    y_center = std::min(std::max(0, y_center), image_height - 1);
    
    x_start = std::min(std::max(0, x_start), image_width - 1);
    x_end   = std::min(std::max(0, x_end), image_width - 1);
    y_start = std::min(std::max(0, y_start), image_height - 1);
    y_end   = std::min(std::max(0, y_end), image_height - 1);
    
    unsigned char color = std::min(std::max(0, int(175 + z*80)), 255);
    
    for(int x = x_start; x<=x_end; ++x) {
        int offset                       = y_center * image_width + x;
        image_rgba[offset]               = {color, 0, color, 0};
    }
    
    for(int y = y_start; y<=y_end; ++y) {
        int offset                       = y * image_width + x_center;
        image_rgba[offset]               = {color, 0, color, 0};
    }
}

}  // namespace TNN_NS
