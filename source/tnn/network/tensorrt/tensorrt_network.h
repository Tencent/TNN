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

#ifndef TNN_SOURCE_TNN_NETWORK_TENSORRT_TENSORRT_NETWORK_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORRT_TENSORRT_NETWORK_H_

#include <iostream>
#include <unordered_set>
#include <cuda_runtime.h>
#include <thread>

#include "tnn/core/default_network.h"
#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"
#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"
#include "tnn/network/tensorrt/tensorrt_tensor.h"
#include "tnn/network/tensorrt/tensorrt_blob_manager.h"

namespace TNN_NS {

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
#ifndef DEBUG
        if (severity == Severity::kINFO || severity == Severity::kVERBOSE) return;
#endif
        const char * skips[] = {
            "INVALID_ARGUMENT: Cannot find binding of given name",
            "Unused Input:",
            "Detected invalid timing cache",
            "unused or used only at compile-time",
        };

        std::string msg_str = std::string(msg);
        for(auto skip : skips) {
            if (msg_str.find(skip) != std::string::npos) {
                return;
            }
        }
        switch (severity) {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: break;
            case Severity::kVERBOSE: std::cerr << "VERBOSE: "; break;
            default: break;
        }
        std::cerr << msg << std::endl;
    }
};

class TensorRTPluginLayerBuilder;

class TensorRTNetwork_ : public DefaultNetwork, public ISharedMemoryChangeListener {
public:
    // @brief TensorRTNetwork_ Constructor
    TensorRTNetwork_();

    // @brief virtual default destructor
    virtual ~TensorRTNetwork_();

    // @brief int net with network config, net structure and net resource info
    // @param config network config info
    // @param net_structure network structure info
    // @param net_resource network resource info
    // @param inputs_shape_map modify input shape, if empty, it will use the
    // shape in proto
    virtual Status Init(NetworkConfig &net_config, ModelConfig &model_config,
        AbstractModelInterpreter* interpreter, InputShapesMap min_inputs_shape,
        InputShapesMap max_inputs_shape, bool enable_const_folder);

    // @brief network forward
    virtual Status Forward();

    // @brief reshape with input shape info
    // @inputs input shape info
    virtual Status Reshape(const InputShapesMap &inputs);

    // @brief tnn instance network infer, it will not wait
    virtual Status ForwardAsync(Callback call_back);

    // @brief OnSharedForwardMemoryChanged for share memory change observer
    virtual void OnSharedForwardMemoryChanged(void *memory);

    // @brief get network forward for all blob memory size
    virtual Status GetForwardMemorySize(int &memory_size);

    // @brief set forward memory when share memory mode is set from external
    virtual Status SetForwardMemory(void *memory);

    static std::unordered_map<std::string, TensorRTPluginLayerBuilder*> GetPluginLayerNameMap();

    std::string GetCacheFileName(std::vector<std::string> params_md5, BlobMap input_map,
        BlobMap output_map, const InputShapesMap &min_inputs_shape, int device_id,
        bool int8_mode, bool use_fp16, bool enable_const_folder);

    std::set<std::string> m_concat_blob_names;

private:
    virtual Status InitLayers(NetStructure *net_structure, NetResource *net_resource, bool enable_const_folder);

    bool IsBlobUsed(Blob* blob);

    Status InitWithoutCache(BlobMap &inputs, BlobMap &outputs, std::string cache_file_name,
        NetResource *net_resource, const InputShapesMap &min_inputs_shape);

    Status CreateExecuteContext();

    Status ReshapeLayers();

    Status DumpAllOutputBlob();

    Status CheckConstBlobs();

    bool int8_mode;
    bool test_mode;
    int m_max_batchsize;
    nvinfer1::ICudaEngine* m_trt_engine;
    nvinfer1::IExecutionContext* m_trt_context;
    TRTLogger m_trt_logger;
    std::unordered_map<std::string, std::shared_ptr<nvinfer1::ITensor>> m_blob_tensor_map;
    std::unordered_set<nvinfer1::ITensor *> m_tensor_set;
    void** m_trt_bindings;
    void* m_context_memory;
    NetResource *net_resource_;
    int device_id_;
    size_t context_memory_size_;

    std::thread::id init_thread_id_;

    std::vector<std::string> const_input_blobs_;
    std::vector<std::string> const_weight_blobs_;

    static std::unordered_map<std::string, TensorRTPluginLayerBuilder*> m_plugin_layer_name_map;
    static std::mutex network_mutex;
};

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORRT_TENSORRT_NETWORK_H_
