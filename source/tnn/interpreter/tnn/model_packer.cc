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
    return g_version_magic_number_v2;
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
        LOGE("invalid proto file name! (%s)\n", file_path.c_str());
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

    // 2nd line: "input_name size n c h w date_type : input_name size n c h w data_type ... ,"
    write_stream << "\"";
    int input_count = net_struc->inputs_shape_map.size();
    int idx         = 0;
    for (auto input_shape : net_struc->inputs_shape_map) {
        write_stream << input_shape.first << " ";
        const auto& input_dims = input_shape.second;
        if (magic_number == g_version_magic_number_v2){
            write_stream << input_dims.size() << " ";
        }
        for (auto item : input_shape.second) {
            write_stream << item << " ";
        }
        if (magic_number == g_version_magic_number_v2) {
            const auto& input_data_type_map = net_struc->input_data_type_map;
            if (input_data_type_map.find(input_shape.first) != input_data_type_map.end()) {
                write_stream << input_data_type_map.find(input_shape.first)->second << " ";
            } else {
                // default data type: float
                write_stream << "0" << " ";
            }
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
            if (layer_type_str.compare(0, 9, "Quantized") != 0) {
                layer_type_str = "Quantized" + layer_type_str;
            }
        }
        // add an identifier to the dynamic range quantization layer
        if (item->param->dynamic_range_quantized) {
            if (layer_type_str.compare(0, 21, "DynamicRangeQuantized") != 0) {
                layer_type_str = "DynamicRangeQuantized" + layer_type_str;
            }
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
        LOGE("invalid model file name! (%s)\n", file_path.c_str());
        return Status(TNNERR_PACK_MODEL, "model file cannot be written");
    }
    auto magic_number = GetMagicNumber();
    if (magic_number > 0) {
        write_stream.write(reinterpret_cast<char *>(&magic_number), sizeof(uint32_t));
    }

    res_header header;
    header.layer_cnt_ = 0;

    int resource_count = 0;
    auto serializer    = GetSerializer(write_stream);
    auto ret           = PackLayers(serializer, false, resource_count);
    if (ret != TNN_OK) {
        write_stream.close();
        return ret;
    }

    header.layer_cnt_ = resource_count;
    if (header.layer_cnt_ < 0) {
        return Status(TNNERR_INVALID_MODEL, "invalid model: layer count is less than 1");
    }
    header.serialize(*serializer);

    ret = PackLayers(serializer, true, resource_count);
    if (ret != TNN_OK) {
        write_stream.close();
        return ret;
    }
    
    // save const_map
    auto const_map = net_resource->constant_map;
    if (const_map.size() > 0) {
        // write magic num
        serializer->PutInt(magic_number);
        // write const map size
        serializer->PutInt((int)const_map.size());
        for (const auto& iter : const_map) {
            serializer->PutString(iter.first);
            serializer->PutRaw(*(iter.second.get()));
        }
    }
    
    write_stream.close();
    if (ret != TNN_OK) {
        return ret;
    }

    return TNN_OK;
}

Status ModelPacker::PackLayers(std::shared_ptr<Serializer> &serializer, bool save_resource, int &resource_count) {
    resource_count = 0;

    NetResource *net_resource = GetNetResource();
    NetStructure *net_struct  = GetNetStructure();

    auto &layer_interpreter_map = ModelInterpreter::GetLayerInterpreterMap();
    auto layers                 = net_struct->layers;
    auto resource_map           = net_resource->resource_map;

    std::set<std::string> blob_scale_set;
    Status result;
    for (const auto &layer_info : layers) {
        // save input blobs scale
        std::string layer_name = layer_info->name;
        if (layer_info->param->quantized) {
            for (auto &input_name : layer_info->inputs) {
                auto blob_scale_name = input_name + BLOB_SCALE_SUFFIX;
                if (blob_scale_set.find(blob_scale_name) != blob_scale_set.end()) {
                    continue;
                }
                if (resource_map.find(blob_scale_name) == resource_map.end() ||
                    resource_map.find(blob_scale_name)->second == nullptr) {
                    continue;
                }
                if (save_resource) {
                    result = PackResource(resource_map, blob_scale_name, serializer);
                    if (result != TNN_OK) {
                        return result;
                    }
                }
                resource_count++;
                blob_scale_set.insert(blob_scale_name);
            }
        }
        // save layer resource
        if (resource_map.find(layer_name) != resource_map.end() && resource_map.find(layer_name)->second != nullptr) {
            if (save_resource) {
                result = PackResource(resource_map, layer_name, serializer);
                if (result != TNN_OK) {
                    return result;
                }
            }
            resource_count++;
        }
        // save output blob scale
        if (layer_info->param->quantized) {
            for (auto &output_name : layer_info->outputs) {
                auto blob_scale_name = output_name + BLOB_SCALE_SUFFIX;
                if (blob_scale_set.find(blob_scale_name) != blob_scale_set.end()) {
                    continue;
                }
                if (resource_map.find(blob_scale_name) == resource_map.end() ||
                    resource_map.find(blob_scale_name)->second == nullptr) {
                    continue;
                }
                if (save_resource) {
                    result = PackResource(resource_map, blob_scale_name, serializer);
                    if (result != TNN_OK) {
                        return result;
                    }
                }
                resource_count++;
                blob_scale_set.insert(blob_scale_name);
            }
        }
    }
    return TNN_OK;
}

Status ModelPacker::PackResource(std::map<std::string, std::shared_ptr<LayerResource>> &resource_map,
                                 std::string &layer_name, std::shared_ptr<Serializer> &serializer) {
    // quantized
    auto &layer_interpreter_map = ModelInterpreter::GetLayerInterpreterMap();
    auto iter                   = resource_map.find(layer_name);
    auto layer_info             = FindLayerInfo(layer_name);
    layer_header ly_header;
    ly_header.name_                = iter->first;
    ly_header.type_                = layer_info->type;
    ly_header.type_str_            = layer_info->type_str;
    static int resource_pack_count = 0;
    ly_header.serialize(*serializer);
    LayerResource *layer_resource = iter->second.get();
    auto layer_interpreter        = layer_interpreter_map[layer_info->type];
    if (layer_interpreter != nullptr) {
        Status result = layer_interpreter->SaveResource(*serializer, layer_info->param.get(), layer_resource);
        if (result != TNN_OK) {
            LOGE(
                "Error: layer interpreter save resource failed (name:%s "
                "type_from_str:%s type:%d)\n",
                ly_header.name_.c_str(), ly_header.type_str_.c_str(), ly_header.type_);
            return Status(TNNERR_PACK_MODEL, "model content is invalid");
        }
    } else {
        LOGE(
            "Error: layer interpreter is null (name:%s "
            "type_from_str:%s type:%d)\n",
            ly_header.name_.c_str(), ly_header.type_str_.c_str(), ly_header.type_);
        return Status(TNNERR_PACK_MODEL, "unsupport layer resource type");
    }
    return TNN_OK;
}

}  // namespace TNN_NS
