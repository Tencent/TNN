// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_CREATE_GRAPH_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_CREATE_GRAPH_H_

#include <proto/graph_config.pb.h>
#include <string>
#include "tnn/core/macro.h"
#include "atlas_common_types.h"

namespace TNN_NS {

void AddOutputEngine(hiai::GraphConfig* graph_config, uint32_t engine_id,
                     const std::string& engine_name, AtlasModelConfig m);

void AddDeviceEngine(hiai::GraphConfig* graph_config, uint32_t engine_id,
                     const std::string& engine_name, AtlasModelConfig m);

void AddDvppEngine(hiai::GraphConfig* graph_config, uint32_t engine_id,
                   const std::string& engine_name, AtlasModelConfig m);

void AddConnect(hiai::GraphConfig* graph_config, uint32_t srcId,
                uint32_t srcPort, uint32_t targetId, uint32_t targetPort);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_CREATE_GRAPH_H_
