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

#ifndef TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_NPU_UTILS_H_
#define TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_NPU_UTILS_H_

#include <tnn/core/blob.h>
#include <tnn/interpreter/layer_resource.h>
#include <tnn/interpreter/raw_buffer.h>

#include "graph/op/array_defs.h"
#include "graph/op/const_defs.h"
#include "graph/compatible/all_ops.h"
#include "graph/operator.h"
#include "hiai_ir_build.h"
#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"

namespace TNN_NS {

class NpuUtils {
public:
    static Status CreateInputData(std::shared_ptr<ge::op::Data> &input_data, std::string &input_name,
                                  DimsVector dims_vector);

    static Status CreateAttrValue(shared_ptr<ge::op::Const>& attr_value, ge::Shape shape, RawBuffer &raw_buffer);

    template <class T>
    static Status CreateAttrArray(std::shared_ptr<ge::op::Const> &attr_value, std::vector<T> data,
                                  ge::TensorDesc input_desc, int shape) {
        ge::AttrValue::TENSOR input_size_tensor = std::make_shared<ge::Tensor>(input_desc);
        input_size_tensor->SetData((uint8_t *)data.data(), sizeof(T) * shape);
        attr_value->set_attr_value(input_size_tensor);
        return TNN_OK;
    }

    static Status WriteModelFile(domi::ModelBufferData &model_buffer_data, std::string file_path);

    static Status CalculateBroadcastSize(vector<int> &weight_shape, EltwiseLayerResource *layer_res,
                                         vector<int> &input_shape);
    static std::string GetFileHash(ModelConfig &model_config);

    static bool FileExits(string model_path);

    static Status GetPadMode(int &pad_mode, int pad_type);

    static int checkNpuVersion(const char *version);

    static std::string modifyModelInputSize(InputShapesMap &inputs_shape, InputShapesMap &instance_input_shapes_map);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_NPU_UTILS_H_
