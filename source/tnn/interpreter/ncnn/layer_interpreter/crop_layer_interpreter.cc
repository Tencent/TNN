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

#include <algorithm>

#include "tnn/interpreter/ncnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/ncnn/ncnn_layer_type.h"
#include "tnn/interpreter/ncnn/ncnn_param_utils.h"
namespace TNN_NS {

namespace ncnn {

    DECLARE_LAYER_INTERPRETER(Crop);

    REGISTER_LAYER_INTERPRETER(Crop, Crop);

    Status CropLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                LayerParam** param) {
        type = ConvertNCNNLayerType(type_name);

        StrideSliceLayerParam* layer_param = new StrideSliceLayerParam();
        *param                             = layer_param;

        auto& p = param_dict;

        int woffset  = GetInt(p, 0, 0);
        int hoffset  = GetInt(p, 1, 0);
        int coffset  = GetInt(p, 2, 0);
        int outw     = GetInt(p, 3, 0);
        int outh     = GetInt(p, 4, 0);
        int outc     = GetInt(p, 5, 0);
        int woffset2 = GetInt(p, 6, 0);
        int hoffset2 = GetInt(p, 7, 0);
        int coffset2 = GetInt(p, 8, 0);

        layer_param->begins = GetIntList(p, 9);
        layer_param->ends   = GetIntList(p, 10);

        std::vector<int> stride_one = {1, 1, 1, 1};
        layer_param->strides        = stride_one;

        bool not_numpy_style_crop = layer_param->begins.size() == 0 && layer_param->ends.size() == 0;

        if (not_numpy_style_crop) {
            int dims = int(HasField(p, 0)) + int(HasField(p, 1)) + int(HasField(p, 2));

            // w h c n
            if (dims == 1) {
                layer_param->begins = {0, 0, woffset, 0};
                layer_param->ends   = {0, 0, -woffset2, 0};
            } else if (dims == 2) {
                layer_param->begins = {0, hoffset, woffset, 0};
                layer_param->ends   = {0, -hoffset2, -woffset2, 0};
            } else if (dims == 3) {
                layer_param->begins = {woffset, hoffset, coffset, 0};
                layer_param->ends   = {-woffset2, -hoffset2, -coffset2, 0};
            } else {
                return Status(TNNERR_INVALID_NETCFG, "ncnn crop layer invalid dims.");
            }
        } else {
            std::reverse(layer_param->begins.begin(), layer_param->begins.end());
            std::reverse(layer_param->ends.begin(), layer_param->ends.end());
        }

        if (layer_param->begins.size() != 4 || layer_param->ends.size() != 4) {
            // TODO fully support crop layer.
            // onnx2ncnn failed to convert onnx slice layer now
            return Status(TNNERR_INVALID_NETCFG, "ncnn crop layer not fully supported now");
        }

        return TNN_OK;
    }

    Status CropLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                   LayerResource** resource) {
        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
