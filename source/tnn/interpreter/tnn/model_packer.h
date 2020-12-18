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

#ifndef TNN_SOURCE_TNN_INTERPRETER_TNN_TNN_MODEL_PACKER_H_
#define TNN_SOURCE_TNN_INTERPRETER_TNN_TNN_MODEL_PACKER_H_

#include "tnn/interpreter/default_model_packer.h"
#include "tnn/interpreter/tnn/objseri.h"

using namespace TNN_NS;
namespace TNN_NS {

// @brief ModelPacker used to save raidnet v1 model
class ModelPacker : public DefaultModelPacker {
public:
    ModelPacker(NetStructure *net_struct, NetResource *net_res)
        : DefaultModelPacker(net_struct, net_res), model_version_(1) {}
    // @brief save the rpn model into files
    virtual Status Pack(std::string proto_path, std::string model_path);

    // @brief set the model version to pack
    void SetVersion(int version);

private:
    std::shared_ptr<LayerInfo> FindLayerInfo(std::string layer_name);
    Status PackProto(std::string file_path);
    Status PackModel(std::string file_path);
    Status PackLayers(std::shared_ptr<Serializer> &serializer, bool save_resource, int &resource_count);
    Status PackResource(std::map<std::string, std::shared_ptr<LayerResource>> &resource_map, std::string &layer_name,
                        std::shared_ptr<Serializer> &serializer);

protected:
    int model_version_ = 1;

    virtual std::string Transfer(std::string content);
    virtual uint32_t GetMagicNumber();
    virtual std::shared_ptr<TNN_NS::Serializer> GetSerializer(std::ostream &os);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_TNN_TNN_MODEL_PACKER_H_
