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

#ifndef TNN_SOURCE_TNN_UTILS_NPU_COMMON_UTILS_H_
#define TNN_SOURCE_TNN_UTILS_NPU_COMMON_UTILS_H_

#include <tnn/core/blob.h>
#include <tnn/interpreter/layer_resource.h>
#include <tnn/interpreter/raw_buffer.h>

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"

namespace TNN_NS {

class NpuCommonUtils {
public:
    static Status CalculateBroadcastSize(std::vector<int> &weight_shape, EltwiseLayerResource *layer_res,
                                         std::vector<int> &input_shape);
    static std::string GetFileHash(ModelConfig &model_config);

    static bool FileExits(std::string model_path);

    static std::string modifyModelInputSize(InputShapesMap &inputs_shape, InputShapesMap &instance_input_shapes_map);

    static Status CalculateOutputShape(LayerType type, std::vector<Blob *> &input_blobs,
                                       std::vector<Blob *> &output_blobs, LayerParam *param, LayerResource *resource,
                                       std::vector<std::string> &outputs_name,
                                       std::vector<std::vector<int>> &output_shapes);

    static Status CreateBlobs(std::vector<BlobDesc> blob_descs, std::vector<Blob *> &blobs);

    static Status ReleaseBlobs(std::vector<Blob *> &input_blobs, std::vector<Blob *> &output_blobs);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_NPU_COMMON_UTILS_H_
