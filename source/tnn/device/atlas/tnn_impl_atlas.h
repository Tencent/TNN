// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_TNN_IMPL_ATLAS_H_
#define TNN_SOURCE_DEVICE_ATLAS_TNN_IMPL_ATLAS_H_

#include "tnn/core/macro.h"
#include "tnn/core/tnn_impl.h"

namespace TNN_NS {

// @brief tnn impl with interpreter
class TNNImplAtlas : public TNNImpl {
public:
    // @brief tnn constructor
    TNNImplAtlas();

    // @brief tnn destructor
    virtual ~TNNImplAtlas();

    // @brief init the tnn, contruct model interpreter
    // @param config config model type and params
    // @return status code: Successful, returns zero. Otherwise, returns
    // error code.
    virtual Status Init(ModelConfig& config);

    // @brief release model interpreter
    virtual Status DeInit();

    //@brief Adds output to the layer. If layerName not found, then search
    // outputIndex.
    //@param output_name Name of the output blob
    //@param output_index Index of the output layer
    //@return status code: If successful, returns zero. Otherwise, returns
    // error
    // code.
    virtual Status AddOutput(const std::string& output_name, int output_index = 0);

    // @brief create an instance
    // @param instance: The instance to be created.
    // @param inputs_shape: modify input shape, or it will use the shape in the
    // proto
    // @param status code: If successful, returns zero. Otherwise, returns
    // error code.
    virtual std::shared_ptr<Instance> CreateInst(NetworkConfig& config, Status& status,
                                                 InputShapesMap inputs_shape = InputShapesMap());

private:
    std::shared_ptr<AbstractModelInterpreter> interpreter_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_TNN_IMPL_ATLAS_H_
