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

#include "tnn/interpreter/tnn/model_packer.h"

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/tnn/model_interpreter.h"
#include "tnn/interpreter/tnn/objseri.h"

namespace TNN_NS {

std::string ModelPacker::Transfer(std::string content) {
    return content;
}

uint32_t ModelPacker::GetMagicNumber() {
    return g_version_magic_number;
}

std::shared_ptr<Serializer> ModelPacker::GetSerializer(std::ostream &os) {
    return std::make_shared<Serializer>(os);
}

Status ModelPacker::Pack(std::string proto_path, std::string model_path) {
    Status ret = TNN_OK;
    ret        = PackProto(proto_path);
    if (ret != TNN_OK) {
        LOGE("Pack TNN Prototxt failed!\n");
        return ret;
    }
    ret = PackModel(model_path);
    if (ret != TNN_OK) {
        LOGE("Pack TNN Model failed!\n");
        return ret;
    }
    return TNN_OK;
}

void ModelPacker::SetVersion(int version) {
    model_version_ = version;
}

std::shared_ptr<LayerInfo> ModelPacker::FindLayerInfo(std::string layer_name) {
    std::shared_ptr<LayerInfo> layer_info;

    if (layer_name.rfind(BLOB_SCALE_SUFFIX) != std::string::npos) {
        // blob scale layer
        layer_info           = std::make_shared<LayerInfo>();
        layer_info->type     = LAYER_BLOB_SCALE;
        layer_info->type_str = "BlobScale";
        layer_info->name     = layer_name;
    } else {
        layer_info = GetLayerInfoFromName(GetNetStructure(), layer_name);
    }

    return layer_info;
}

Status ModelPacker::PackProto(std::string file_path) {
    Status ret              = TNN_OK;
    NetStructure *net_struc = GetNetStructure();

    std::ofstream write_stream;
    write_stream.open(file_path);
    if (!write_stream || !write_stream.is_open() || !write_stream.good()) {
        write_stream.close();
        return Status(TNNERR_PACK_MODEL, "proto file cannot be written");
    }

    // 1st line: "1 <blob_size> 1 <magic_num> ,"
    auto magic_number = GetMagicNumber();
    if (magic_number > 0) {
        write_stream << "\"1 " << net_struc->blobs.size() << " 1 " << magic_number << " ,\"" << std::endl;
    } else {
        write_stream << "\"1 " << net_struc->blobs.size() << " 1 "
                     << ",\"" << std::endl;
    }

    // 2nd line: "<input_name1> <d0> <d1> ... : <input_name2> ... ,"
    write_stream << "\"";
    int input_count = net_struc->inputs_shape_map.size();
    int idx         = 0;
    for (auto input_shape : net_struc->inputs_shape_map) {
        write_stream << input_shape.first << " ";
        for (auto item : input_shape.second) {
            write_stream << item << " ";
        }

        if (input_count > 1 && idx < (input_count - 1)) {
            write_stream << ": ";
        }
        idx++;
    }
    write_stream << ",\"" << std::endl;

    // 3rd line: all blobs  " <blob1> <blob2> ... ,"
    write_stream << "\" ";
    for (auto item : net_struc->blobs) {
        write_stream << item << " ";
    }
    write_stream << ",\"" << std::endl;

    // 4th line: "<output_blob1> <output_blob2> .., ,"
    write_stream << "\"";
    for (auto item : net_struc->outputs) {
        write_stream << item << " ";
    }
    write_stream << ",\"" << std::endl;

    // 5th line: " <layer_count> ,"
    write_stream << "\" " << net_struc->layers.size() << " ,\"" << std::endl;

    // each layer info
    auto &layer_interpreter_map = ModelInterpreter::GetLayerInterpreterMap();
    for (auto item : net_struc->layers) {
        write_stream << "\"";
        // layer type
        std::string layer_type_str = item->type_str;
        if (item->param->quantized) {
            layer_type_str = "Quantized" + layer_type_str;
        }
        layer_type_str = Transfer(layer_type_str);
        write_stream << layer_type_str << " ";

        // layer name
        std::string layer_name = item->name;
        layer_name             = Transfer(layer_name);
        write_stream << layer_name << " ";

        // input/output size
        write_stream << item->inputs.size() << " " << item->outputs.size() << " ";
        // input name
        for (auto name : item->inputs) {
            std::string input_name = name;
            input_name             = Transfer(input_name);
            write_stream << input_name << " ";
        }

        // output name
        for (auto name : item->outputs) {
            std::string output_name = name;
            output_name             = Transfer(output_name);
            write_stream << output_name << " ";
        }

        auto layer_interpreter = layer_interpreter_map[item->type];
        if (layer_interpreter != nullptr) {
            layer_interpreter->SaveProto(write_stream, item->param.get());
        }

        write_stream << ",\"" << std::endl;
    }

    write_stream.close();

    return TNN_OK;
}

Status ModelPacker::PackModel(std::string file_path) {
    NetResource *net_resource = GetNetResource();
    NetStructure *net_struct  = GetNetStructure();
    std::ofstream write_stream;
    write_stream.open(file_path, std::ios::binary);
    if (!write_stream || !write_stream.is_open() || !write_stream.good()) {
        write_stream.close();
        return Status(TNNERR_PACK_MODEL, "model file cannot be written");
    }

    auto magic_number = GetMagicNumber();
    if (magic_number > 0) {
        write_stream.write(reinterpret_cast<char *>(&magic_number), sizeof(uint32_t));
    }

    res_header header;
    header.layer_cnt_ = 0;
    for (auto item : net_resource->resource_map) {
        if (item.second != nullptr) {
            header.layer_cnt_++;
        }
    }
    if (header.layer_cnt_ <= 0) {
        return Status(TNNERR_INVALID_MODEL, "invalid model: layer count is less than 1");
    }

    auto serializer = GetSerializer(write_stream);
    header.serialize(*serializer);

    auto &layer_interpreter_map = ModelInterpreter::GetLayerInterpreterMap();
    for (auto iter = net_resource->resource_map.begin(); iter != net_resource->resource_map.end(); ++iter) {
        if (iter->second == nullptr) {
            continue;
        }
        layer_header ly_head;
        ly_head.name_   = iter->first;
        auto layer_info = FindLayerInfo(ly_head.name_);
        if (layer_info == nullptr) {
            LOGE("layer_packer: can't find layer(%s) in net struct!\n", ly_head.name_.c_str());
            return Status(TNNERR_PACK_MODEL, "layer_packer: can't find layer in net struct");
        }

        ly_head.type_     = LAYER_NOT_SUPPORT;  // just use type_str_ to judge
        ly_head.type_str_ = layer_info->type_str;
        ly_head.serialize(*serializer);

        LayerResource *layer_resource = iter->second.get();
        auto layer_interpreter        = layer_interpreter_map[layer_info->type];
        if (layer_interpreter != NULL) {
            Status result = layer_interpreter->SaveResource(*serializer, layer_info->param.get(), layer_resource);
            if (result != TNN_OK) {
                write_stream.close();
                return result;
            }
        } else {
            write_stream.close();
            LOGE(
                "Error: layer_interpreter is null (name:%s "
                "type_from_str:%s type:%d)\n",
                ly_head.name_.c_str(), ly_head.type_str_.c_str(), ly_head.type_);
            return Status(TNNERR_LOAD_MODEL, "model content is invalid");
            ;
        }
    }

    write_stream.close();

    return TNN_OK;
}

}  // namespace TNN_NS
