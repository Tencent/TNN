// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_DSP_SNPE_NETWORK_H_
#define TNN_SOURCE_DEVICE_DSP_SNPE_NETWORK_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "SNPE/SNPE.hpp"
#include "core/abstract_network.h"

namespace TNN_NS {

class SnpeNetwork : public AbstractNetwork {
public:
    // @brief virtual default destructor
    virtual ~SnpeNetwork();

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
    std::unique_ptr<zdl::SNPE::SNPE> snpe_;
    zdl::DlSystem::UserBufferMap input_map_;
    zdl::DlSystem::UserBufferMap output_map_;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>
        snpe_userbacked_input_buffers_;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>
        snpe_userbacked_output_buffers_;
    std::unordered_map<std::string, std::vector<uint8_t>>
        application_input_buffers_;
    std::unordered_map<std::string, std::vector<uint8_t>>
        application_output_buffers_;

    BlobMap input_blob_map_;
    BlobMap output_blob_map_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_DSP_SNPE_NETWORK_H_
