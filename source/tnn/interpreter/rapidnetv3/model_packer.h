// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_RAPIDNETV3_MODEL_PACKER_H_
#define TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_RAPIDNETV3_MODEL_PACKER_H_

#include "tnn/interpreter/tnn/model_packer.h"
#include "tnn/interpreter/rapidnetv3/objseri.h"

namespace rapidnetv3 {

    // @brief ModelPacker used to save raidnet v1 model
    class ModelPacker : public TNN_NS::ModelPacker {
    public:
        ModelPacker(NetStructure *net_struct, NetResource *net_res)
        : TNN_NS::ModelPacker(net_struct, net_res) {
            model_version_ = MV_RPNV3;
        }
        
        // @brief save the rpn model into files
        Status Pack(std::string proto_path, std::string model_path) override;

    protected:
        std::string Transfer(std::string content) override;
        uint32_t GetMagicNumber() override;
        std::shared_ptr<TNN_NS::Serializer> GetSerializer(std::ostream &os) override;
    };

}  // namespace rapidnetv3



#endif  // TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_RAPIDNETV3_MODEL_PACKER_H_
