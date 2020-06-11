// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_NETWORK_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_NETWORK_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "atlas_common_types.h"
#include "atlas_data_recv.h"
#include "hiaiengine/api.h"
#include "tnn/core/macro.h"
#include "tnn/core/abstract_network.h"

namespace TNN_NS {

class AtlasNetwork : public AbstractNetwork {
public:
    // @brief virtual default destructor
    virtual ~AtlasNetwork();

    // @brief init network with net cfg and net res.
    // @param net_cfg
    // @param net_res
    virtual Status Init(NetworkConfig &net_config, ModelConfig &model_config,
                        AbstractModelInterpreter *interpreter,
                        InputShapesMap inputs_shape);

    // @brief deinit release init create resource
    virtual Status DeInit();

    //  @brief return the amount of memory required for forward
    //  @param memory_size: the memory size used by tnn layers for
    //  forward
    //  @return error code: If successful, returns zero. Otherwise, returns
    //  an error code.
    virtual Status GetForwardMemorySize(int &memory_size);

    //  @brief: set memory used by the tnn instance without forward
    //  memory, the memory size must be at least that returned by
    //  GetForwardMemorySize(). releasing or otherwise using the memory for
    //  other purposes during the tnn network run will result in
    //  undefined behavior.
    //  @param memory: the memory used by tnn layers for forward
    //  @return error code: If successful, returns zero. Otherwise, returns
    //  an error code.
    //
    virtual Status SetForwardMemory(void *memory);

    // @brief network infer
    virtual Status Reshape(const InputShapesMap &inputs);

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void **command_queue);

    // @brief network infer, it will sync to wait result
    virtual Status Forward();

    // @brief tnn instance network infer, it will not wait
    virtual Status ForwardAsync(Callback call_back);

    // @brief get all input blobs
    // @param blobs input blobs name map
    virtual Status GetAllInputBlobs(BlobMap &blobs);

    // @brief get all output blobs
    // @param blobs output blobs name map
    virtual Status GetAllOutputBlobs(BlobMap &blobs);

private:
    // @brief if device is disconnected, call this func
    static HIAI_StatusT DeviceDisconnectCallBack();

    // @brief allocate input buffer on host
    Status AllocateInputBufferForAtlas();

    // @brief allocate output buffer on host
    Status AllocateOutputBufferForAtlas();

    // @brief send data to dvpp engine to use dvpp and aipp
    Status SendInputToDvppEngine();

    // @brief send data to inference engine directly
    Status SendInputToInferenceEngine();

    std::shared_ptr<hiai::Graph> graph_;
    AtlasModelConfig atlas_config_;
    std::shared_ptr<AtlasDataRecv> data_recv_;
    std::shared_ptr<TransferDataType> input_trans_data_;
    std::shared_ptr<DvppTransDataType> input_dvpp_data_;
    std::map<std::string, std::shared_ptr<uint8_t>> input_data_buffer_;
    std::map<std::string, std::shared_ptr<uint8_t>> output_data_buffer_;
    std::map<std::string, long> output_addr_map_;
    uint32_t batch_id_ = 0;
    uint32_t frame_id_ = 0;

    BlobMap input_blob_map_;
    BlobMap output_blob_map_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_NETWORK_H_
