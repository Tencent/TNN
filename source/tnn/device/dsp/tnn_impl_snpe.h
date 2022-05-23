// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_CORE_TNN_IMPL_SNPE_H_
#define TNN_CORE_TNN_IMPL_SNPE_H_

#include "core/tnn_impl.h"

namespace TNN_NS {

// @brief tnn impl with interpreter
class TNNImplSnpe : public TNNImpl {
public:
    // @brief tnn constructor
    TNNImplSnpe();

    // @brief tnn destructor
    virtual ~TNNImplSnpe();

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
    virtual Status AddOutput(const std::string& output_name,
                             int output_index = 0);

    // @brief create an instance
    // @param instance: The instance to be created.
    // @param inputs_shape: modify input shape, or it will use the shape in the
    // proto
    // @param status code: If successful, returns zero. Otherwise, returns
    // error code.
    virtual std::shared_ptr<Instance> CreateInst(
        NetworkConfig& config, Status& status,
        InputShapesMap inputs_shape = InputShapesMap());

private:
    std::shared_ptr<AbstractModelInterpreter> interpreter_;
};

}  // namespace TNN_NS

#endif  // TNN_CORE_TNN_IMPL_SNPE_H_
