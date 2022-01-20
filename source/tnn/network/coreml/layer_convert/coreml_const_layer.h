
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

#include "coreml_base_layer.h"

namespace TNN_NS {

class CoreMLConstLayer : public CoreMLBaseLayer {
public:
   CoreMLConstLayer(LayerType layer_type) : CoreMLBaseLayer(layer_type){};
   virtual ~CoreMLConstLayer() {};
   Status Init(std::string output_name ,RawBuffer raw_buffer);
   
   virtual std::string GetLayerName();
   
   virtual Status BuildLayerType();
   virtual Status BuildLayerParam();
   virtual Status BuildConstantWeightsLayer();
   virtual std::vector<std::string> BuildLayerInputs();
   virtual std::vector<std::string> BuildLayerOutputs();
   
protected:
   std::string output_name_;
   RawBuffer raw_buffer_;
   shared_ptr<uint64_t> shape_;
   shared_ptr<CoreML__Specification__WeightParams> weight_param_;
    
private:
    RawBuffer cvt_raw_buffer_;
};

}  // namespace TNN_NS
