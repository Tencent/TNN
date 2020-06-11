// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_create_graph.h"
#include <stdio.h>

namespace TNN_NS {

void AddOutputEngine(hiai::GraphConfig* graph_config, uint32_t engine_id,
                     const std::string& engine_name, AtlasModelConfig m) {
    hiai::EngineConfig* engine = graph_config->add_engines();
    engine->set_id(engine_id);
    engine->set_engine_name(engine_name);
    engine->set_side(hiai::EngineConfig_RunSide_HOST);
    engine->set_thread_num(1);
    engine->add_so_name("./libOutputEngine.so");
}

void AddDeviceEngine(hiai::GraphConfig* graph_config, uint32_t engine_id,
                     const std::string& engine_name, AtlasModelConfig m) {
    hiai::EngineConfig* engine = graph_config->add_engines();
    engine->set_id(engine_id);
    engine->set_engine_name(engine_name);
    engine->set_side(hiai::EngineConfig_RunSide_DEVICE);
    engine->set_thread_num(1);
    engine->add_so_name("./libInferenceEngine.so");

    hiai::AIConfig* ai_conf  = engine->mutable_ai_config();
    hiai::AIConfigItem* item = ai_conf->add_items();
    item->set_name("model_path");
    item->set_value(m.om_path);

    item = ai_conf->add_items();
    item->set_name("dynamic_aipp");
    item->set_value(m.dynamic_aipp ? "1" : "0");

    item = ai_conf->add_items();
    item->set_name("daipp_swap_rb");
    item->set_value(m.daipp_swap_rb ? "1" : "0");

    item = ai_conf->add_items();
    item->set_name("daipp_norm");
    item->set_value(m.daipp_norm ? "1" : "0");
}

void AddDvppEngine(hiai::GraphConfig* graph_config, uint32_t engine_id,
                   const std::string& engine_name, AtlasModelConfig m) {
    hiai::EngineConfig* engine = graph_config->add_engines();
    engine->set_id(engine_id);
    engine->set_engine_name(engine_name);
    engine->set_side(hiai::EngineConfig_RunSide_DEVICE);
    engine->set_thread_num(1);
    engine->add_so_name("./libDvppEngine.so");

    char temp[32];
    hiai::AIConfig* ai_conf = engine->mutable_ai_config();

    hiai::AIConfigItem* item = ai_conf->add_items();
    item->set_name("userHome");
    item->set_value("/home/atlas300/tools");

    item = ai_conf->add_items();
    item->set_name("self_crop");
    item->set_value("1");

    item = ai_conf->add_items();
    item->set_name("point_x");
    item->set_value("-1");

    item = ai_conf->add_items();
    item->set_name("dump_value");
    item->set_value("0");

    item = ai_conf->add_items();
    item->set_name("resize_width");
    sprintf(temp, "%d", m.width);
    item->set_value(temp);

    item = ai_conf->add_items();
    item->set_name("resize_height");
    sprintf(temp, "%d", m.height);
    item->set_value(temp);
}

void AddConnect(hiai::GraphConfig* graph_config, uint32_t srcId,
                uint32_t srcPort, uint32_t targetId, uint32_t targetPort) {
    hiai::ConnectConfig* connect = graph_config->add_connects();
    connect->set_src_engine_id(srcId);
    connect->set_src_port_id(srcPort);
    connect->set_target_engine_id(targetId);
    connect->set_target_port_id(targetPort);
}

}  // namespace TNN_NS
