// Copyright 2019 Tencent. All Rights Reserved

#include "tnn/interpreter/rapidnetv3/model_packer.h"

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/rapidnetv3/objseri.h"
#include "tnn/interpreter/rapidnetv3/model_interpreter.h"



namespace rapidnetv3 {
    std::string ModelPacker::Transfer(std::string content) {
        if (MV_RPNV3 == model_version_) {
            content = BlurMix(content.c_str(), (int)content.length(), true);
        }
        return content;
    }

    uint32_t ModelPacker::GetMagicNumber() {
        if (MV_TNN == model_version_) {
            return g_version_magic_number_tnn;
        } else if (MV_RPNV3 == model_version_) {
            return g_version_magic_number_rapidnet_v3;
        } else if (MV_TNN_V2 == model_version_) {
            return g_version_magic_number_tnn_v2;
        } else {
            return 0;
        }
    }

    std::shared_ptr<TNN_NS::Serializer> ModelPacker::GetSerializer(std::ostream &os) {
        auto ser = std::make_shared<rapidnetv3::Serializer>(os, model_version_);
        return ser;
    }

    Status ModelPacker::Pack(std::string proto_path,
                                       std::string model_path) {
        return TNN_NS::ModelPacker::Pack(proto_path, model_path);
    }

}  // namespace rapidnetv3


