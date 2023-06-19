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

#include "tnn/optimizer/net_optimizer_fuse_matmul_concat.h"

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

namespace optimizer {

    NetOptimizerRegister<NetOptimizerFuseMatmulConcat> g_net_optimizer_fuse_matmul_concat(OptPriority::P1);

    std::string NetOptimizerFuseMatmulConcat::Strategy() {
        return kNetOptimizerFuseMatmulConcat;
    }

    bool NetOptimizerFuseMatmulConcat::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        if (device == DEVICE_CUDA) {
            return true;
        }
        return false;
    }

    class BaseMatmulCombiner {
        public:
            BaseMatmulCombiner() {}
            virtual ~BaseMatmulCombiner() {}

            Status Combine(NetStructure *structure, NetResource *resource);

        protected:
            virtual std::vector<LayerType> GetPattern() = 0;
            virtual std::set<std::string> CombineLayers(const std::vector<std::string> &concat_inputs) = 0;

            MatMulLayerResource *CheckAndGetMatmulResource(std::shared_ptr<LayerInfo> mm_layer, bool is_first_layer);
            EltwiseLayerResource *CheckAndGetAddResource(std::shared_ptr<LayerInfo> add_layer, bool is_first_layer);
            InnerProductLayerResource *CheckAndGetIpResource(std::shared_ptr<LayerInfo> ip_layer, InnerProductLayerParam *first_ip_param, bool is_first_layer);
            Status ConcatMatmulResource(std::vector<MatMulLayerResource*> &mm_resources);
            Status ConcatAddResource(std::vector<EltwiseLayerResource*> &add_resources);
            Status ConcatIpResource(std::vector<InnerProductLayerResource*> &ip_resources);

            NetStructure *structure_;
            NetResource *resource_;
            std::vector<std::shared_ptr<LayerInfo>> layers_orig_;
            int layers_count_;
            std::unordered_map<std::string, int> blob_to_layerid_;
            std::unordered_map<std::string, int> blob_to_usecount_;
            int gemm_k_;
            int gemm_n_;
            bool has_bias_;
            int bias_n_;

        private:
            Status Init(NetStructure *structure, NetResource *resource);
            bool IsPatternMatches(const std::vector<std::string> &inputs, const std::vector<LayerType> &pattern);
    };

    Status NetOptimizerFuseMatmulConcat::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }
        if (!resource) {
            LOGE("Error: empty NetResource\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetResource");
        }

        for (const auto &combiner : GetCombiners()) {
            RETURN_ON_FAIL(combiner->Combine(structure, resource));
        }

        return TNN_OK;
    }

    class MatmulAddSigmoidCombiner : public BaseMatmulCombiner {
        public:
            MatmulAddSigmoidCombiner() : BaseMatmulCombiner() {}
            virtual ~MatmulAddSigmoidCombiner() {}
        private:
            virtual std::vector<LayerType> GetPattern() override;
            virtual std::set<std::string> CombineLayers(const std::vector<std::string> &concat_inputs) override;
    };

    class InnerproductSigmoidCombiner : public BaseMatmulCombiner {
        public:
            InnerproductSigmoidCombiner() : BaseMatmulCombiner() {}
            virtual ~InnerproductSigmoidCombiner() {}
        private:
            virtual std::vector<LayerType> GetPattern() override;
            virtual std::set<std::string> CombineLayers(const std::vector<std::string> &concat_inputs) override;
    };

    std::vector<std::shared_ptr<BaseMatmulCombiner>> NetOptimizerFuseMatmulConcat::GetCombiners() {
        std::vector<std::shared_ptr<BaseMatmulCombiner>> res;
        res.push_back(std::make_shared<MatmulAddSigmoidCombiner>());
        res.push_back(std::make_shared<InnerproductSigmoidCombiner>());
        return res;
    }

    Status BaseMatmulCombiner::Init(NetStructure *structure, NetResource *resource) {
        structure_ = structure;
        resource_  = resource;
        layers_orig_  = structure_->layers;
        layers_count_ = layers_orig_.size();
        for (int index = 0; index < layers_count_; index++) {
            for (const auto &in : layers_orig_[index]->inputs) {
                blob_to_usecount_[in]++;
            }
            for (const auto &out : layers_orig_[index]->outputs) {
                blob_to_layerid_[out] = index;
            }
        }
        for (const auto &out : structure_->outputs) {
            blob_to_usecount_[out]++;
        }
        return TNN_OK;
    }

    bool BaseMatmulCombiner::IsPatternMatches(const std::vector<std::string> &inputs,
                                              const std::vector<LayerType> &pattern) {
        std::string in_blob;
        for (const auto &input : inputs) {
            auto blob = input;
            for (auto iter = pattern.rbegin(); iter != pattern.rend(); ++iter) {
                if (blob_to_usecount_.find(blob) == blob_to_usecount_.end() ||
                    blob_to_usecount_.at(blob) != 1) {
                    return false;
                }
                if (blob_to_layerid_.find(blob) == blob_to_layerid_.end()) {
                    return false;
                }
                auto prev_layer = structure_->layers[blob_to_layerid_.at(blob)];
                if (prev_layer->type != *iter ||
                    prev_layer->inputs.size() != 1 ||
                    prev_layer->outputs.size() != 1) {
                    return false;
                }
                blob = prev_layer->inputs[0];
            }
            if (in_blob == "") {
                in_blob = blob;
            } else {
                if (blob != in_blob) {
                    return false;
                }
            }
        }
        return true;
    }

    Status BaseMatmulCombiner::Combine(NetStructure *structure, NetResource *resource) {
        RETURN_ON_FAIL(Init(structure, resource));

        std::set<std::string> remove_layers;
        for (int index = 0; index < layers_count_; index++) {
            auto concat_layer = layers_orig_[index];
            if (concat_layer->type != LAYER_CONCAT || concat_layer->inputs.size() <= 1 || concat_layer->outputs.size() != 1) {
                continue;
            }

            if (!IsPatternMatches(concat_layer->inputs, GetPattern())) {
                continue;
            }

            std::set<std::string> cur_remove_layers = CombineLayers(concat_layer->inputs);
            if (!cur_remove_layers.empty()) {
                for (const auto &r : cur_remove_layers) {
                    remove_layers.insert(r);
                }
                auto prev_layer = layers_orig_[blob_to_layerid_[concat_layer->inputs[0]]];
                prev_layer->outputs = {concat_layer->outputs[0]};
                remove_layers.insert(concat_layer->name);
            }
        }

        if (remove_layers.empty()) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_optimized;
        for (int index = 0; index < layers_count_; index++) {
            if (remove_layers.find(layers_orig_[index]->name) == remove_layers.end()) {
                layers_optimized.push_back(layers_orig_[index]);
            }
        }
        structure_->layers = layers_optimized;

        return TNN_OK;
    }

    MatMulLayerResource *BaseMatmulCombiner::CheckAndGetMatmulResource(std::shared_ptr<LayerInfo> mm_layer, bool is_first_layer) {
        MatMulLayerParam* mm_param = dynamic_cast<MatMulLayerParam*>(mm_layer->param.get());
        if (!mm_param || mm_param->weight_position != 1) {
            return nullptr;
        }
        if (resource_->resource_map.find(mm_layer->name) == resource_->resource_map.end()) {
            return nullptr;
        }
        MatMulLayerResource* mm_res = dynamic_cast<MatMulLayerResource*>(resource_->resource_map.at(mm_layer->name).get());
        if (!mm_res || mm_res->weight.GetBufferDims().size() != 2) {
            return nullptr;
        }
        if (is_first_layer) {
            gemm_k_ = mm_res->weight.GetBufferDims()[0];
            gemm_n_ = mm_res->weight.GetBufferDims()[1];
        } else {
            if (mm_res->weight.GetBufferDims()[0] != gemm_k_) {
                return nullptr;
            }
            gemm_n_ += mm_res->weight.GetBufferDims()[1];
        }
        return mm_res;
    }

    EltwiseLayerResource *BaseMatmulCombiner::CheckAndGetAddResource(std::shared_ptr<LayerInfo> add_layer, bool is_first_layer) {
        MultidirBroadcastLayerParam* add_param = dynamic_cast<MultidirBroadcastLayerParam*>(add_layer->param.get());
        if (!add_param || add_param->weight_input_index != 1) {
            return nullptr;
        }

        if (resource_->resource_map.find(add_layer->name) == resource_->resource_map.end()) {
            return nullptr;
        }
        EltwiseLayerResource* add_res = dynamic_cast<EltwiseLayerResource*>(resource_->resource_map.at(add_layer->name).get());
        if (!add_res || add_res->element_handle.GetBufferDims().size() != 1) {
            return nullptr;
        }
        if (is_first_layer) {
            bias_n_ = add_res->element_handle.GetBufferDims()[0];
        } else {
            bias_n_ += add_res->element_handle.GetBufferDims()[0];
        }
        if (bias_n_ != gemm_n_) {
            return nullptr;
        }
        return add_res;
    }

    InnerProductLayerResource *BaseMatmulCombiner::CheckAndGetIpResource(std::shared_ptr<LayerInfo> ip_layer, InnerProductLayerParam* first_ip_param, bool is_first_layer) {
        InnerProductLayerParam* ip_param = dynamic_cast<InnerProductLayerParam*>(ip_layer->param.get());
        if (!ip_param ||
            ip_param->axis != first_ip_param->axis ||
            ip_param->has_bias != first_ip_param->has_bias ||
            ip_param->transpose != 0) {
            return nullptr;
        }

        if (resource_->resource_map.find(ip_layer->name) == resource_->resource_map.end()) {
            return nullptr;
        }
        InnerProductLayerResource* ip_res = dynamic_cast<InnerProductLayerResource*>(resource_->resource_map.at(ip_layer->name).get());
        if (!ip_res ||
            ip_res->weight_handle.GetBufferDims().size() != 2) {
            return nullptr;
        }
        if (is_first_layer) {
            has_bias_ = ip_param->has_bias;
            gemm_k_ = ip_res->weight_handle.GetBufferDims()[0];
            gemm_n_ = ip_res->weight_handle.GetBufferDims()[1];
            if (ip_param->has_bias) {
                bias_n_ = ip_res->bias_handle.GetBufferDims()[0];
            }
        } else {
            if (ip_res->weight_handle.GetBufferDims()[0] != gemm_k_) {
                return nullptr;
            }
            gemm_n_ += ip_res->weight_handle.GetBufferDims()[1];
            if (ip_param->has_bias) {
                bias_n_ += ip_res->bias_handle.GetBufferDims()[0];
            }
        }
        if (ip_param->has_bias && bias_n_ != gemm_n_) {
            return nullptr;
        }
        return ip_res;
    }

    Status BaseMatmulCombiner::ConcatMatmulResource(std::vector<MatMulLayerResource*> &mm_resources) {
        auto dtype = mm_resources[0]->weight.GetDataType();

        int dsize = DataTypeUtils::GetBytesSize(dtype);
        RawBuffer weight_handle(gemm_k_ * gemm_n_ * dsize);
        weight_handle.SetBufferDims({gemm_k_, gemm_n_});
        weight_handle.SetDataType(dtype);

        auto weight_start = weight_handle.force_to<char*>();
        for (const auto &res : mm_resources) {
            int cur_gemm_n = res->weight.GetBufferDims()[1];
            for (int k = 0; k < gemm_k_; ++k) {
                memcpy(weight_start + k * gemm_n_ * dsize, res->weight.force_to<char*>() + k * cur_gemm_n * dsize, cur_gemm_n * dsize);
            }
            weight_start += cur_gemm_n * dsize;
        }

        mm_resources[0]->weight = weight_handle;
        return TNN_OK;
    }

    Status BaseMatmulCombiner::ConcatAddResource(std::vector<EltwiseLayerResource*> &add_resources) {
        auto dtype = add_resources[0]->element_handle.GetDataType();

        int dsize = DataTypeUtils::GetBytesSize(dtype);
        RawBuffer bias_handle(gemm_n_ * dsize);
        bias_handle.SetBufferDims({gemm_n_});
        bias_handle.SetDataType(dtype);

        auto bias_start = bias_handle.force_to<char*>();
        for (const auto &res : add_resources) {
            memcpy(bias_start, res->element_handle.force_to<char*>(), res->element_handle.GetBytesSize());
            bias_start += res->element_handle.GetBytesSize();
        }

        add_resources[0]->element_handle = bias_handle;
        add_resources[0]->element_shape = bias_handle.GetBufferDims();
        return TNN_OK;
    }

    Status BaseMatmulCombiner::ConcatIpResource(std::vector<InnerProductLayerResource*> &ip_resources) {
        auto dtype = ip_resources[0]->weight_handle.GetDataType();

        int dsize = DataTypeUtils::GetBytesSize(dtype);
        RawBuffer weight_handle(gemm_k_ * gemm_n_ * dsize);
        weight_handle.SetBufferDims({gemm_k_, gemm_n_});
        weight_handle.SetDataType(dtype);
        RawBuffer bias_handle(gemm_n_ * dsize);
        bias_handle.SetBufferDims({gemm_n_});
        bias_handle.SetDataType(dtype);

        auto weight_start = weight_handle.force_to<char*>();
        auto bias_start = bias_handle.force_to<char*>();
        for (const auto &res : ip_resources) {
            memcpy(weight_start, res->weight_handle.force_to<char*>(), res->weight_handle.GetBytesSize());
            weight_start += res->weight_handle.GetBytesSize();
            if (has_bias_) {
                memcpy(bias_start, res->bias_handle.force_to<char*>(), res->bias_handle.GetBytesSize());
                bias_start += res->bias_handle.GetBytesSize();
            }
        }

        ip_resources[0]->weight_handle = weight_handle;
        if (has_bias_) {
            ip_resources[0]->bias_handle = bias_handle;
        }
        return TNN_OK;
    }

    std::vector<LayerType> MatmulAddSigmoidCombiner::GetPattern() {
        return {LAYER_MATMUL, LAYER_ADD, LAYER_SIGMOID};
    }

    std::set<std::string> MatmulAddSigmoidCombiner::CombineLayers(const std::vector<std::string> &concat_inputs) {
        auto sigmoid_layer = layers_orig_[blob_to_layerid_[concat_inputs[0]]];
        auto add_layer     = layers_orig_[blob_to_layerid_[sigmoid_layer->inputs[0]]];
        auto matmul_layer  = layers_orig_[blob_to_layerid_[add_layer->inputs[0]]];

        MatMulLayerResource* mm_res = CheckAndGetMatmulResource(matmul_layer, true);
        if (!mm_res) {
            return {};
        }
        EltwiseLayerResource* add_res = CheckAndGetAddResource(add_layer, true);
        if (!add_res) {
            return {};
        }

        std::set<std::string> remove_layers;
        std::vector<MatMulLayerResource*> mm_resources = {mm_res};
        std::vector<EltwiseLayerResource*> add_resources = {add_res};
        for (int i = 1; i < concat_inputs.size(); ++i) {
            auto sigmoid_layer = layers_orig_[blob_to_layerid_[concat_inputs[i]]];
            auto add_layer     = layers_orig_[blob_to_layerid_[sigmoid_layer->inputs[0]]];
            auto matmul_layer  = layers_orig_[blob_to_layerid_[add_layer->inputs[0]]];

            MatMulLayerResource* mm_res = CheckAndGetMatmulResource(matmul_layer, false);
            if (!mm_res) {
                return {};
            }
            EltwiseLayerResource* add_res = CheckAndGetAddResource(add_layer, false);
            if (!add_res) {
                return {};
            }

            mm_resources.push_back(mm_res);
            add_resources.push_back(add_res);
            remove_layers.insert(sigmoid_layer->name);
            remove_layers.insert(add_layer->name);
            remove_layers.insert(matmul_layer->name);
        }

        if (!remove_layers.empty()) {
            if (ConcatMatmulResource(mm_resources) != TNN_OK) {
                return {};
            }
            if (!ConcatAddResource(add_resources) != TNN_OK) {
                return {};
            }
        }
        return remove_layers;
    }

    std::vector<LayerType> InnerproductSigmoidCombiner::GetPattern() {
        return {LAYER_INNER_PRODUCT, LAYER_SIGMOID};
    }

    std::set<std::string> InnerproductSigmoidCombiner::CombineLayers(const std::vector<std::string> &concat_inputs) {
        auto sigmoid_layer = layers_orig_[blob_to_layerid_[concat_inputs[0]]];
        auto ip_layer      = layers_orig_[blob_to_layerid_[sigmoid_layer->inputs[0]]];

        InnerProductLayerParam* ip_param = dynamic_cast<InnerProductLayerParam*>(ip_layer->param.get());

        InnerProductLayerResource* ip_res = CheckAndGetIpResource(ip_layer, ip_param, true);
        if (!ip_res) {
            return {};
        }

        std::set<std::string> remove_layers;
        std::vector<InnerProductLayerResource*> ip_resources = {ip_res};
        for (int i = 1; i < concat_inputs.size(); ++i) {
            auto sigmoid_layer = layers_orig_[blob_to_layerid_[concat_inputs[i]]];
            auto ip_layer      = layers_orig_[blob_to_layerid_[sigmoid_layer->inputs[0]]];

            InnerProductLayerResource* ip_res = CheckAndGetIpResource(ip_layer, ip_param, false);
            if (!ip_res) {
                return {};
            }

            ip_resources.push_back(ip_res);
            remove_layers.insert(sigmoid_layer->name);
            remove_layers.insert(ip_layer->name);
        }

        if (!remove_layers.empty()) {
            if (ConcatIpResource(ip_resources) != TNN_OK) {
                return {};
            }
            ip_param->num_output = gemm_n_;
        }
        return remove_layers;
    }

}  // namespace optimizer

}  // namespace TNN_NS
