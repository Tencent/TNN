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

#include "tnn/utils/npu_common_utils.h"

#include "tnn/core/macro.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

Status NpuCommonUtils::CalculateBroadcastSize(std::vector<int> &weight, EltwiseLayerResource *layer_res,
                                              std::vector<int> &input) {
    int input_count = DimsVectorUtils::Count(input, 1);
    if (weight.size() < 4) {
        weight             = {1, 1, 1, 1};
        int layer_res_size = layer_res->element_handle.GetDataCount();
        if (layer_res_size == 1) {
            // single element
            weight[1] = layer_res_size;
        } else if (layer_res_size == input[1]) {
            // channel broadcast
            weight[1] = layer_res_size;
        } else if (layer_res_size == input_count) {
            // element broadcast
            weight[1] = input[1];
            weight[2] = input[2];
            weight[3] = input[3];
        } else if (layer_res_size == input[3]) {
            weight[3] = input[3];
        } else {
            return Status(TNNERR_LAYER_ERR, "Error: unsupported broadcast type");
        }
        layer_res->element_shape = weight;
    }
    return TNN_OK;
}

std::string NpuCommonUtils::GetFileHash(ModelConfig &model_config) {
    std::string file_content = model_config.params[1] + model_config.params[0];
    int hash                 = 0;
    for (size_t i = 0; i < file_content.length(); ++i)
        hash = 65599 * hash + file_content.at(i);
    return ToString(hash ^ (hash >> 16));
}

bool NpuCommonUtils::FileExits(std::string model_path) {
    std::ifstream infile(model_path);
    return infile.good();
}

Status NpuCommonUtils::CalculateOutputShape(LayerType type, std::vector<Blob *> &input_blobs,
                                            std::vector<Blob *> &output_blobs, LayerParam *param,
                                            LayerResource *resource, std::vector<std::string> &outputs_name,
                                            std::vector<std::vector<int>> &output_shapes) {
    BaseLayer *shape_calculator = CreateLayer(type);

    Status ret = shape_calculator->InferShapeAhead(input_blobs, output_blobs, param, resource);
    RETURN_ON_NEQ(ret, TNN_OK);

    for (int i = 0; i < outputs_name.size(); i++) {
        output_shapes.push_back(output_blobs[i]->GetBlobDesc().dims);
    }

    delete (shape_calculator);

    return TNN_OK;
}

Status NpuCommonUtils::CreateBlobs(std::vector<BlobDesc> blob_descs, std::vector<Blob *> &blobs) {
    for (auto &desc : blob_descs) {
        Blob *blob = new Blob(desc);
        blobs.push_back(blob);
    }

    return TNN_OK;
}

Status NpuCommonUtils::ReleaseBlobs(std::vector<Blob *> &input_blobs, std::vector<Blob *> &output_blobs) {
    for (auto &blob : input_blobs) {
        delete (blob);
    }
    for (auto &blob : output_blobs) {
        delete (blob);
    }
    input_blobs.clear();
    output_blobs.clear();

    return TNN_OK;
}

}  // namespace TNN_NS
