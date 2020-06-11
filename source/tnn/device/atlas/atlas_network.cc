// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_network.h"
#include <time.h>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include "atlas_common_types.h"
#include "atlas_create_graph.h"
#include "atlas_model_interpreter.h"
#include "atlas_utils.h"
#include "hiaiengine/ai_memory.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<AtlasNetwork>>
    g_network_impl_atlas_factory_register(NETWORK_TYPE_ATLAS);

std::mutex g_mtx_output;

AtlasNetwork::~AtlasNetwork() {
    DeInit();
}

Status AtlasNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config,
                          AbstractModelInterpreter *interpreter,
                          InputShapesMap inputs_shape) {
    AtlasModelInterpreter *atlas_interpreter =
        dynamic_cast<AtlasModelInterpreter *>(interpreter);

    atlas_config_ = atlas_interpreter->GetModelConfig();

    // Init HIAI
    HIAI_StatusT status = HIAI_Init(net_config.device_id);
    if (status != HIAI_OK) {
        LOGE("Init HIAI failed!\n");
        return TNNERR_ATLAS_RUNTIME_ERROR;
    }

    // Create Graph
    std::shared_ptr<hiai::GraphConfigList> glist(new hiai::GraphConfigList());
    hiai::GraphConfig *graph_config = glist->add_graphs();
    graph_config->set_graph_id(atlas_config_.graph_id);
    graph_config->set_priority(0);

    AddDeviceEngine(graph_config, atlas_config_.inference_engine_id,
                    "InferenceEngine", atlas_config_);
    AddOutputEngine(graph_config, atlas_config_.output_engine_id,
                    "OutputEngine", atlas_config_);

    if (atlas_config_.with_dvpp == true) {
        AddDvppEngine(graph_config, atlas_config_.dvpp_engine_id, "DvppEngine",
                      atlas_config_);
        AddConnect(graph_config, atlas_config_.dvpp_engine_id, 0,
                   atlas_config_.inference_engine_id, 0);
        AddConnect(graph_config, atlas_config_.inference_engine_id, 0,
                   atlas_config_.output_engine_id, 0);
    } else {
        AddConnect(graph_config, atlas_config_.inference_engine_id, 0,
                   atlas_config_.output_engine_id, 0);
    }
    status = hiai::Graph::CreateGraph(*glist);

    if (status != HIAI_OK) {
        LOGE("Fail to create graph!\n");
        return TNNERR_ATLAS_GRAPH_INIT_ERROR;
    }

    // Register recv function
    graph_ = hiai::Graph::GetInstance(atlas_config_.graph_id);
    if (graph_ == nullptr) {
        LOGE("Fail to get the graph\n");
        return TNNERR_ATLAS_RUNTIME_ERROR;
    }

    hiai::EnginePortID target_port_config;
    target_port_config.graph_id  = atlas_config_.graph_id;
    target_port_config.engine_id = atlas_config_.output_engine_id;
    target_port_config.port_id   = 0;

    data_recv_.reset(new AtlasDataRecv());
    graph_->SetDataRecvFunctor(target_port_config, data_recv_);
    graph_->RegisterEventHandle(hiai::HIAI_DEVICE_DISCONNECT_EVENT,
                                DeviceDisconnectCallBack);

    // allocate input and output buffer
    input_data_buffer_.clear();
    output_data_buffer_.clear();
    Status ret = AllocateInputBufferForAtlas();
    if (ret != TNN_OK)
        return ret;
    ret = AllocateOutputBufferForAtlas();
    if (ret != TNN_OK)
        return ret;

    Blob *input_blob   = input_blob_map_.begin()->second;
    int bytes_per_elem = 1;
    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        bytes_per_elem = 4;
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        bytes_per_elem = 2;
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        bytes_per_elem = 1;
    }
    int size_in_bytes =
        DimsVectorUtils::Count(input_blob->GetBlobDesc().dims) * bytes_per_elem;

    // input_trans_data_ init
    input_trans_data_                  = std::make_shared<TransferDataType>();
    input_trans_data_->info.cmd_type   = CT_DataTransfer;
    input_trans_data_->info.query_type = QT_None;
    input_trans_data_->info.dim_info.batch = input_blob->GetBlobDesc().dims[0];
    input_trans_data_->info.dim_info.channel =
        input_blob->GetBlobDesc().dims[1];
    input_trans_data_->info.dim_info.height = input_blob->GetBlobDesc().dims[2];
    input_trans_data_->info.dim_info.width  = input_blob->GetBlobDesc().dims[3];
    input_trans_data_->info.size_in_bytes   = size_in_bytes;
    input_trans_data_->output_info.output_map = output_addr_map_;
    input_trans_data_->data_len               = size_in_bytes;
    input_trans_data_->data = input_data_buffer_.begin()->second;

    // input_dvpp_data_ init
    input_dvpp_data_ = std::make_shared<DvppTransDataType>();
    input_dvpp_data_->output_info.output_map = output_addr_map_;
    input_dvpp_data_->b_info.batch_size = input_blob->GetBlobDesc().dims[0];
    input_dvpp_data_->img_data.format   = (hiai::IMAGEFORMAT)IMAGE_TYPE_JPEG;
    input_dvpp_data_->img_data.channel  = input_blob->GetBlobDesc().dims[1];
    input_dvpp_data_->img_data.height   = input_blob->GetBlobDesc().dims[2];
    input_dvpp_data_->img_data.width    = input_blob->GetBlobDesc().dims[3];
    // Note: img_data.size need to be decide in forward by
    // BlobHandle.bytes_offset
    input_dvpp_data_->img_data.size = 0;
    input_dvpp_data_->img_data.data = input_data_buffer_.begin()->second;

    return TNN_OK;
}

Status AtlasNetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return TNN_OK;
}

Status AtlasNetwork::SetForwardMemory(void *memory) {
    return TNN_OK;
}

Status AtlasNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status AtlasNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

HIAI_StatusT AtlasNetwork::DeviceDisconnectCallBack() {
    return HIAI_OK;
}

Status AtlasNetwork::AllocateInputBufferForAtlas() {
    hiai::EnginePortID engine_id;
    engine_id.graph_id  = atlas_config_.graph_id;
    engine_id.engine_id = atlas_config_.inference_engine_id;
    engine_id.port_id   = 0;

    uint8_t *buffer = nullptr;
    int buffer_size = 16;
    HIAI_StatusT get_ret =
        hiai::HIAIMemory::HIAI_DMalloc(buffer_size, (void *&)buffer);
    if (HIAI_OK != get_ret || nullptr == buffer) {
        LOGE("DMalloc buffer error (size=%d) (ret=0x%x)!\n", buffer_size,
             get_ret);
        return TNNERR_ATLAS_MALLOC_ERROR;
    }
    std::shared_ptr<TransferDataType> trans_data =
        std::make_shared<TransferDataType>();
    trans_data->info.cmd_type   = CT_InfoQuery;
    trans_data->info.query_type = QT_InputDims;
    trans_data->data_len        = buffer_size;
    trans_data->data.reset(buffer, [](uint8_t *) {});

    graph_->SendData(engine_id, "TransferDataType",
                     std::static_pointer_cast<void>(trans_data));

    if (data_recv_->WaitInputBlobMap(5000) != 0) {
        LOGE("Get Input Blobmap timeout!\n");
        return TNNERR_ATLAS_TIMEOUT_ERROR;
    }

    input_blob_map_ = data_recv_->GetInputBlobMap();

    // allocate input buffer
    for (auto item = input_blob_map_.begin(); item != input_blob_map_.end();
         ++item) {
        int bytes_per_elem = 1;
        if (item->second->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            bytes_per_elem = 4;
        } else if (item->second->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            bytes_per_elem = 2;
        } else if (item->second->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            bytes_per_elem = 1;
        }
        int size_in_bytes =
            DimsVectorUtils::Count(item->second->GetBlobDesc().dims) *
            bytes_per_elem;
        uint8_t *buffer      = nullptr;
        HIAI_StatusT get_ret = hiai::HIAIMemory::HIAI_DMalloc(
            size_in_bytes, (void *&)buffer, hiai::MALLOC_DEFAULT_TIME_OUT,
            hiai::HIAI_MEMORY_ATTR_MANUAL_FREE);
        if (HIAI_OK != get_ret || nullptr == buffer) {
            LOGE("DMalloc buffer error (size=%d)(ret=0x%x)!\n", size_in_bytes,
                 get_ret);
            return TNNERR_ATLAS_MALLOC_ERROR;
        }
        input_data_buffer_[item->first] =
            std::shared_ptr<uint8_t>(buffer, hiai::HIAIMemory::HIAI_DFree);
        BlobHandle blob_handle;
        blob_handle.base = buffer;
        item->second->SetHandle(blob_handle);
    }

    return TNN_OK;
}

Status AtlasNetwork::AllocateOutputBufferForAtlas() {
    hiai::EnginePortID engine_id;
    engine_id.graph_id  = atlas_config_.graph_id;
    engine_id.engine_id = atlas_config_.inference_engine_id;
    engine_id.port_id   = 0;

    uint8_t *buffer = nullptr;
    int buffer_size = 16;
    HIAI_StatusT get_ret =
        hiai::HIAIMemory::HIAI_DMalloc(buffer_size, (void *&)buffer);
    if (HIAI_OK != get_ret || nullptr == buffer) {
        LOGE("DMalloc buffer error (size=%d) (ret=0x%x)!\n", buffer_size,
             get_ret);
        return TNNERR_ATLAS_MALLOC_ERROR;
    }
    std::shared_ptr<TransferDataType> trans_data =
        std::make_shared<TransferDataType>();
    trans_data->info.cmd_type   = CT_InfoQuery;
    trans_data->info.query_type = QT_OutputDims;
    trans_data->data_len        = buffer_size;
    trans_data->data.reset(buffer, [](uint8_t *) {});

    graph_->SendData(engine_id, "TransferDataType",
                     std::static_pointer_cast<void>(trans_data));

    if (data_recv_->WaitOutputBlobMap(5000) != 0) {
        LOGE("Get Output Blobmap timeout!\n");
        return TNNERR_ATLAS_TIMEOUT_ERROR;
    }

    output_blob_map_ = data_recv_->GetOutputBlobMap();

    // allocate output buffer
    for (auto item = output_blob_map_.begin(); item != output_blob_map_.end();
         ++item) {
        int bytes_per_elem = 1;
        if (item->second->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            bytes_per_elem = 4;
        } else if (item->second->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            bytes_per_elem = 2;
        } else if (item->second->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            bytes_per_elem = 1;
        }
        int size_in_bytes =
            DimsVectorUtils::Count(item->second->GetBlobDesc().dims) *
            bytes_per_elem;
        uint8_t *buffer      = nullptr;
        HIAI_StatusT get_ret = hiai::HIAIMemory::HIAI_DMalloc(
            size_in_bytes, (void *&)buffer, hiai::MALLOC_DEFAULT_TIME_OUT,
            hiai::HIAI_MEMORY_ATTR_MANUAL_FREE);
        if (HIAI_OK != get_ret || nullptr == buffer) {
            LOGE("DMalloc buffer error (size=%d)(ret=0x%x)!\n", size_in_bytes,
                 get_ret);
            return TNNERR_ATLAS_MALLOC_ERROR;
        }
        output_data_buffer_[item->first] =
            std::shared_ptr<uint8_t>(buffer, hiai::HIAIMemory::HIAI_DFree);
        output_addr_map_[item->first] = reinterpret_cast<long>(buffer);

        BlobHandle blob_handle;
        blob_handle.base = buffer;
        item->second->SetHandle(blob_handle);
    }

    return TNN_OK;
}

Status AtlasNetwork::SendInputToDvppEngine() {
    hiai::EnginePortID engine_id;
    engine_id.graph_id  = atlas_config_.graph_id;
    engine_id.engine_id = atlas_config_.dvpp_engine_id;
    engine_id.port_id   = 0;

    // assign b_info: batch_ID, frame_ID
    input_dvpp_data_->b_info.batch_ID = batch_id_;
    input_dvpp_data_->b_info.frame_ID.clear();
    input_dvpp_data_->b_info.frame_ID.push_back(frame_id_);

    // assign img_data: size
    Blob *input_blob                = input_blob_map_.begin()->second;
    input_dvpp_data_->img_data.size = input_blob->GetHandle().bytes_offset;

    if (frame_id_ == input_dvpp_data_->b_info.batch_size - 1) {
        // the last frame in one batch
        // assign output_info: output_cv_addr, time_s, time_ns
        std::shared_ptr<std::condition_variable> cv =
            std::make_shared<std::condition_variable>();
        std::unique_lock<std::mutex> lck(g_mtx_output);
        input_dvpp_data_->output_info.output_cv_addr =
            reinterpret_cast<long>(cv.get());

        struct timespec time_stamp;
        clock_gettime(CLOCK_MONOTONIC, &time_stamp);
        input_dvpp_data_->output_info.time_s  = (long)time_stamp.tv_sec;
        input_dvpp_data_->output_info.time_ns = (long)time_stamp.tv_nsec;

        graph_->SendData(engine_id, "DvppTransDataType",
                         std::static_pointer_cast<void>(input_dvpp_data_));

        if (cv->wait_for(lck, std::chrono::milliseconds(FORWARD_TIMEOUT)) ==
            std::cv_status::timeout) {
            LOGE("forward timeout!\n");
            return TNNERR_ATLAS_TIMEOUT_ERROR;
        }

        batch_id_++;
    } else {
        graph_->SendData(engine_id, "DvppTransDataType",
                         std::static_pointer_cast<void>(input_dvpp_data_));
    }

    frame_id_ = (++frame_id_) % input_dvpp_data_->b_info.batch_size;

    return TNN_OK;
}

Status AtlasNetwork::SendInputToInferenceEngine() {
    hiai::EnginePortID engine_id;
    engine_id.graph_id  = atlas_config_.graph_id;
    engine_id.engine_id = atlas_config_.inference_engine_id;
    engine_id.port_id   = 0;

    // assign output_info
    std::shared_ptr<std::condition_variable> cv =
        std::make_shared<std::condition_variable>();
    std::unique_lock<std::mutex> lck(g_mtx_output);
    input_trans_data_->output_info.output_cv_addr =
        reinterpret_cast<long>(cv.get());

    struct timespec time_stamp;
    clock_gettime(CLOCK_MONOTONIC, &time_stamp);
    input_trans_data_->output_info.time_s  = (long)time_stamp.tv_sec;
    input_trans_data_->output_info.time_ns = (long)time_stamp.tv_nsec;

    graph_->SendData(engine_id, "TransferDataType",
                     std::static_pointer_cast<void>(input_trans_data_));

    if (cv->wait_for(lck, std::chrono::milliseconds(FORWARD_TIMEOUT)) ==
        std::cv_status::timeout) {
        LOGE("forward timeout!\n");
        return TNNERR_ATLAS_TIMEOUT_ERROR;
    }

    return TNN_OK;
}

Status AtlasNetwork::Reshape(const InputShapesMap &inputs) {
    LOGD("Atlas Reshape!\n");
    return TNN_OK;
}

Status AtlasNetwork::DeInit() {
    for (auto item : input_blob_map_) {
        delete item.second;
    }
    input_blob_map_.clear();
    for (auto item : output_blob_map_) {
        delete item.second;
    }
    output_blob_map_.clear();

    input_trans_data_.reset();
    input_dvpp_data_.reset();
    data_recv_.reset();
    input_data_buffer_.clear();
    output_data_buffer_.clear();
    hiai::Graph::DestroyGraph(atlas_config_.graph_id);
    return TNN_OK;
}

Status AtlasNetwork::GetCommandQueue(void **command_queue) {
    return TNN_OK;
}

Status AtlasNetwork::Forward() {
    LOGD("Atlas Forward!\n");

    Status ret = TNN_OK;

    if (atlas_config_.with_dvpp == true) {
        ret = SendInputToDvppEngine();
    } else {
        ret = SendInputToInferenceEngine();
    }

    return ret;
}

Status AtlasNetwork::ForwardAsync(Callback call_back) {
    LOGD("Atlas Async Forward! (as same as Forward by now)\n");
    return Forward();
}

}  // namespace TNN_NS
