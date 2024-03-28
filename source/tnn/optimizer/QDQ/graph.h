#ifndef TNN_SOURCE_TNN_NET_OPTIMIZER_QDQ_GRAPH_H_
#define TNN_SOURCE_TNN_NET_OPTIMIZER_QDQ_GRAPH_H_

#include <stdint.h>
#include <memory>
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_NS {

namespace QDQ {
typedef int32_t layer_id_t;

static const uint32_t INVALID_NODEID = INT32_MAX;

class Graph {
public:
    Graph(NetStructure *net_structure, NetResource *net_resource)
        : net_structure_(net_structure), net_resource_(net_resource) {}
    ~Graph() {}

    std::vector<layer_id_t> FindPredecessors(layer_id_t) const;

    std::vector<layer_id_t> FindSuccessors(layer_id_t) const;

    std::shared_ptr<LayerInfo> GetLayerById(layer_id_t);

    std::shared_ptr<LayerInfo> CloneLayerById(layer_id_t, int times);

    Status ReplaceWithLayer(layer_id_t, std::shared_ptr<LayerInfo>);

    Status InsertLayers(layer_id_t, std::vector<std::shared_ptr<LayerInfo>> &);

    Status EraseLayerById(layer_id_t);

    Status EliminateDeadLayer();

    layer_id_t GetMaxLayerId();

    Status SetLayerResByName(const std::string &, std::shared_ptr<LayerResource>);

    std::shared_ptr<LayerResource> GetLayerResByName(const std::string &);

    Status SetConstResByName(const std::string &, std::shared_ptr<RawBuffer>);

    std::shared_ptr<RawBuffer> GetConstResByName(const std::string &);

    layer_id_t GetProducerByName(const std::string &) const;

    std::vector<layer_id_t> GetConsumerByName(const std::string &blob_name) const;

    layer_id_t FindPos(std::shared_ptr<LayerInfo>) const;

private:
    NetStructure *net_structure_;
    NetResource *net_resource_;
};
}
}

#endif