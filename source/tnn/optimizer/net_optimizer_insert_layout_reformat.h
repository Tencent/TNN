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

#ifndef TNN_SOURCE_TNN_OPTIMIZER_NET_OPTIMIZER_INSERT_LAYOUT_REFORMAT_H_
#define TNN_SOURCE_TNN_OPTIMIZER_NET_OPTIMIZER_INSERT_LAYOUT_REFORMAT_H_

#include <string>

#include "tnn/core/abstract_device.h"
#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/optimizer/net_optimizer.h"

namespace TNN_NS {

namespace optimizer {

    //@brief net optimize: insert reformat layer between layers with different layouts
    class NetOptimizerInsertLayoutReformat : public NetOptimizer {
    public:
        virtual std::string Strategy();
        virtual bool IsSupported(const NetworkConfig& net_config);
        virtual Status Optimize(NetStructure* structure, NetResource* resource);

        Status AdjustLayer(std::vector<std::shared_ptr<LayerInfo>>& layers_orig, NetStructure* structure,
                           const std::set<std::string>& constant_layers, DataFormat cur_layer_layout,
                           DataFormat reformat_layout, std::shared_ptr<LayerInfo>& new_layer,
                           std::vector<std::string>& reformat_outs, const std::string& reformat_name_suffix,
                           const int index, const int count);

    private:
        std::shared_ptr<const ImplementedLayout> GetLayoutsByLayerType(LayerType type);

        AbstractDevice* device_;
        AbstractDevice* adaptor_device_;
        std::map<std::string, DataFormat> layer_choosed_layout;
        const NetworkConfig* net_config_;
    };

}  // namespace optimizer

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_OPTIMIZER_NET_OPTIMIZER_INSERT_LAYOUT_REFORMAT_H_
