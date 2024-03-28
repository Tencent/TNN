#include "tnn/optimizer/QDQ/graph.h"

namespace TNN_NS {
namespace QDQ {

std::vector<layer_id_t> Graph::FindPredecessors(layer_id_t lid) const {
    std::vector<layer_id_t> res;
    auto layer = net_structure_->layers[lid];
    for (auto &input : layer->inputs) {
        auto lid = GetProducerByName(input);
        res.emplace_back(lid);
    }

    return res;
}

std::vector<layer_id_t> Graph::FindSuccessors(layer_id_t lid) const {
    std::vector<layer_id_t> res;
    auto layer = net_structure_->layers[lid];
    for (auto &output : layer->outputs) {
        auto successors = GetConsumerByName(output);
        res.insert(res.end(), successors.begin(), successors.end());
    }

    return res;
}

std::shared_ptr<LayerInfo> Graph::GetLayerById(layer_id_t lid) {
    if (lid == INVALID_NODEID || lid < 0 || lid >= net_structure_->layers.size()) {
        return nullptr;
    } else {
        return net_structure_->layers[lid];
    }
}

Status Graph::ReplaceWithLayer(layer_id_t lid, std::shared_ptr<LayerInfo> new_layer) {
    net_structure_->layers[lid] = new_layer;
    return TNN_OK;
}

Status Graph::EraseLayerById(layer_id_t lid) {
    auto begin = net_structure_->layers.begin();
    net_structure_->layers.erase(begin + lid);
    return TNN_OK;
}

layer_id_t Graph::GetMaxLayerId() {
    return net_structure_->layers.size() - 1;
}

Status Graph::SetLayerResByName(const std::string &name, std::shared_ptr<LayerResource> l_res) {
    net_resource_->resource_map[name] = l_res;
    return TNN_OK;
}

std::shared_ptr<LayerResource> Graph::GetLayerResByName(const std::string &name) {
    if (!net_resource_->resource_map.count(name)) {
        return nullptr;
    } else {
        return net_resource_->resource_map[name];
    }
}

Status Graph::SetConstResByName(const std::string &name, std::shared_ptr<RawBuffer> const_res) {
    net_resource_->constant_map[name] = const_res;
    return TNN_OK;
}

std::shared_ptr<RawBuffer> Graph::GetConstResByName(const std::string &name) {
    if (!net_resource_->constant_map.count(name)) {
        return nullptr;
    } else {
        return net_resource_->constant_map[name];
    }
}

layer_id_t Graph::GetProducerByName(const std::string &blob_name) const {
    for (int l = 0; l < net_structure_->layers.size(); l++) {
        auto layer = net_structure_->layers[l];
        std::set<std::string> output_set(layer->outputs.begin(), layer->outputs.end());
        if (output_set.count(blob_name))
            return l;
    }
    return INVALID_NODEID;
}

std::vector<layer_id_t> Graph::GetConsumerByName(const std::string &blob_name) const {
    std::vector<layer_id_t> res;
    for (int l = 0; l < net_structure_->layers.size(); l++) {
        auto layer = net_structure_->layers[l];
        std::set<std::string> input_set(layer->inputs.begin(), layer->inputs.end());
        if (input_set.count(blob_name))
            res.emplace_back(l);
    }
    return res;
}

layer_id_t Graph::FindPos(std::shared_ptr<LayerInfo> layer) const {
    int res = INVALID_NODEID;
    for (int l = 0; l < net_structure_->layers.size(); l++) {
        if (layer.get() == net_structure_->layers[l].get()) {
            res = l;
            break;
        }
    }
    return res;
}

std::shared_ptr<LayerInfo> Graph::CloneLayerById(layer_id_t lid, int times) {
    auto old_layer = net_structure_->layers[lid];
    auto new_layer = old_layer->Copy();
    new_layer->name = old_layer->name + "_clone_" + std::to_string(times);

    if (net_resource_->resource_map.count(old_layer->name)) {
        net_resource_->resource_map[new_layer->name] = net_resource_->resource_map[old_layer->name];
    }

    std::vector<std::string> new_outputs;
    for (auto &output : old_layer->outputs) {
        auto new_output = output + "_clone_" + std::to_string(times);
        if (net_resource_->constant_map.count(output)) {
            net_resource_->constant_map[new_output] = net_resource_->constant_map[output];
        }
        new_outputs.emplace_back(new_output);
        net_structure_->blobs.insert(new_output);
    }
    new_layer->outputs = new_outputs;

    return new_layer;
}

Status Graph::InsertLayers(layer_id_t lid, std::vector<std::shared_ptr<LayerInfo>> &insert_layers) {
    auto begin = net_structure_->layers.begin();
    net_structure_->layers.insert(begin + lid, insert_layers.begin(), insert_layers.end());
    return TNN_OK;
}

Status Graph::EliminateDeadLayer() {
    std::for_each(net_structure_->layers.begin(), net_structure_->layers.end(), [&](std::shared_ptr<LayerInfo> &l) {
        if (l->type == LAYER_NOT_SUPPORT) {
            if (this->net_resource_->resource_map.count(l->name)) {
                this->net_resource_->resource_map.erase(l->name);
            }
            // Todo : erase const map
        }
    });

    net_structure_->layers.erase(
        std::remove_if(net_structure_->layers.begin(), net_structure_->layers.end(),
                       [](std::shared_ptr<LayerInfo> &l) { return l->type == LAYER_NOT_SUPPORT;}),
        net_structure_->layers.end());

    return TNN_OK;
}
}  // namespace QDQ
}  // namespace TNN_NS