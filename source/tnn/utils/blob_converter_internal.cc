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

#include <mutex>
#include <string>
#include "tnn/utils/blob_converter.h"

#include "tnn/utils/blob_converter_internal.h"
#include "tnn/utils/dims_function_utils.h"

namespace TNN_NS {

bool NeedDoScaleBias(const MatConvertParam& param) {
    for (auto s : param.scale) {
        if (s != 1.0f) {
            return true;
        }
    }
    for (auto b : param.bias) {
        if (b != 0.0f) {
            return true;
        }
    }

    return false;
}

BlobConverter::BlobConverter(Blob* blob) {
    blob_ = blob;
    impl_ = BlobConverterManager::Shared()->CreateBlobConverterAcc(blob);
}

Status BlobConverter::ConvertToMat(Mat& image, MatConvertParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_INIT_LAYER, "image converter is nil, check device type");
    }

    Status ret = CheckScaleBiasInParam(image, param, true);
    if (ret != TNN_OK) {
        return ret;
    }

    return impl_->ConvertToMat(image, param, command_queue);
}

Status BlobConverter::ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_INIT_LAYER, "image converter is nil, check device type");
    }

    Status ret = CheckScaleBiasInParam(image, param, true);
    if (ret != TNN_OK) {
        return ret;
    }

    return impl_->ConvertToMatAsync(image, param, command_queue);
}

Status BlobConverter::ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_INIT_LAYER, "image converter is nil, check device type");
    }

    Status ret = CheckScaleBiasInParam(image, param, false);
    if (ret != TNN_OK) {
        return ret;
    }

    return impl_->ConvertFromMat(image, param, command_queue);
}

Status BlobConverter::ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_INIT_LAYER, "image converter is nil, check device type");
    }

    Status ret = CheckScaleBiasInParam(image, param, false);
    if (ret != TNN_OK) {
        return ret;
    }

    return impl_->ConvertFromMatAsync(image, param, command_queue);
}

Status BlobConverter::CheckScaleBiasInParam(Mat& image, MatConvertParam& param, bool convert_to_mat) {
    int channel = 0;
    if (convert_to_mat) {
        CHECK_PARAM_NULL(blob_);
        channel = DimsFunctionUtils::GetDim(blob_->GetBlobDesc().dims, 1);
    } else {
        channel = image.GetChannel();
    }
    // 非图像类的Mat channel和scale/bias长度与不匹配时，如果scale全1，bias全0，会默认调整，否则报错
    if ((image.GetMatType() == NCHW_FLOAT || image.GetMatType() == RESERVED_BFP16_TEST ||
         image.GetMatType() == RESERVED_FP16_TEST || image.GetMatType() == RESERVED_INT8_TEST ||
         image.GetMatType() == NC_INT32) && (channel > param.scale.size() || channel > param.bias.size())) {
        if (!NeedDoScaleBias(param)) {
            param.scale = std::vector<float>(channel, 1.0f);
            param.bias = std::vector<float>(channel, 0.0f);
        } else {
            LOGE("blob converter param is invalid, scale bias not match Mat channel,"
                 "scale size: %d, bias size: %d, Mat channel: %d\n", (int)param.scale.size(),
                (int)param.bias.size(), image.GetChannel());
            return Status(TNNERR_PARAM_ERR, "blob converter param is invalid!");
        }
    }

    return TNN_OK;
}

std::shared_ptr<BlobConverterManager>& BlobConverterManager::Shared() {
    static std::once_flag once;
    static std::shared_ptr<BlobConverterManager> g_global_blob_converter_manager;
    std::call_once(once, []() { g_global_blob_converter_manager = std::make_shared<BlobConverterManager>(); });
    return g_global_blob_converter_manager;
}

std::shared_ptr<BlobConverterAcc> BlobConverterManager::CreateBlobConverterAcc(Blob* blob) {
    auto iter = converter_creater_map_.find(blob->GetBlobDesc().device_type);
    if (iter != converter_creater_map_.end()) {
        return iter->second->CreateBlobConverterAcc(blob);
    }
    return nullptr;
}

int BlobConverterManager::RegisterBlobConverterAccCreater(DeviceType type,
                                                          std::shared_ptr<BlobConverterAccCreater> creater) {
    auto iter = converter_creater_map_.find(type);
    if (iter != converter_creater_map_.end()) {
        LOGE("Error: device_type(%d) cannot be registered twice\n", type);
        return 1;
    }
    if (!creater) {
        LOGE("Error: MatBlobConverterAccCreater is nil device_type(%d)\n", type);
        return 1;
    }
    converter_creater_map_[type] = creater;
    return 0;
}

BlobConverterManager::BlobConverterManager() {}
BlobConverterManager::~BlobConverterManager() {}

}  // namespace TNN_NS
