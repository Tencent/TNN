//
// Created by 李烨 on 20/7/20.
//
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

#include "graph/attr_value.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

class NpuReduceLayer : public NpuBaseLayer {
public:
    NpuReduceLayer(LayerType layer_type) : NpuBaseLayer(layer_type){};
    virtual ~NpuReduceLayer() {}

protected:
    Status GetReduceParam() {
        // parameter and weight of the pooling layer
        auto param = dynamic_cast<ReduceLayerParam *>(param_);
        CHECK_PARAM_NULL(param);
        axes_ = param->axis;
        std::vector<int> input_shape_vec = input_ops_[0]->GetShape();

        // check if all reduce
        if (param->all_reduce) {
            axes_.clear();
            for (int i = 0; i < input_shape_vec.size(); i++) {
                axes_.push_back(i);
            }
        } else {
            for (int i = 0; i < axes_.size(); i++) {
                if (axes_[i] < 0) {
                    axes_[i] = input_shape_vec.size() + axes_[i];
                }
            }
        }

        keep_dims_ = param->keep_dims;
        return TNN_OK;
    }

    template <class T>
    Status ReduceConvert() {
        GetReduceParam();

        int reduce_size = axes_.size();
        ge::Shape weight_shape({reduce_size});
        ge::TensorDesc desc(weight_shape, ge::FORMAT_NCHW, ge::DT_INT32);
        std::shared_ptr<ge::op::Const> axes_op = std::make_shared<ge::op::Const>(layer_name_ + "_axes");
        NpuUtils::CreateAttrArray(axes_op, axes_, desc, reduce_size);
        weight_ops_.push_back(axes_op);

        auto output = std::make_shared<T>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_input_axes(*axes_op);
        output->set_attr_keep_dims(keep_dims_);
        ADD_OUTPUT_OP(output)
    }

    template <class T>
    Status ReduceConvertAttr() {
        GetReduceParam();

        std::vector<int64_t> axes(axes_.begin(), axes_.end());

        auto output = std::make_shared<T>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_attr_axes(axes);
        output->set_attr_keep_dims(keep_dims_);
        ADD_OUTPUT_OP(output)
    }

private:
    std::vector<shared_ptr<ge::Operator>> weight_ops_;
protected:
    std::vector<int> axes_;
    int keep_dims_;
};

}  // namespace TNN_NS