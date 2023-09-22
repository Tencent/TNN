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

    //@brief get input shapes map from model
    virtual Status GetModelInputShapesMap(InputShapesMap& shapes_map);

    //@brief get input data types map from model
    virtual Status GetModelInputDataTypeMap(InputDataTypeMap& data_type_map);

    //@brief return input names from model
    virtual Status GetModelInputNames(std::vector<std::string>& input_names);

    //@brief return output names from model
    virtual Status GetModelOutputNames(std::vector<std::string>& output_names);

    // @brief create an instance
    // @param instance: The instance to be created.
    // @param inputs_shape: modify input shape, or it will use the shape in the
    // proto
    // @param status code: If successful, returns zero. Otherwise, returns
    // error code.
    virtual std::shared_ptr<Instance> CreateInst(NetworkConfig& config, Status& status,
                                                 InputShapesMap inputs_shape = InputShapesMap(),
                                                 InputDataTypeMap inputs_data_type = InputDataTypeMap());

    // @brief create an instance
    // @param instance: The instance to be created.
    // @param min_inputs_shape: support min shape
    // @param max_inputs_shape: support max shape
    // @param status code: If successful, returns zero. Otherwise, returns
    // error code.
    virtual std::shared_ptr<Instance> CreateInst(NetworkConfig& config, Status& status, InputShapesMap min_inputs_shape,
                                                 InputShapesMap max_inputs_shape, InputDataTypeMap inputs_data_type = InputDataTypeMap());

private:
    std::shared_ptr<AbstractModelInterpreter> interpreter_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_TNN_IMPL_ATLAS_H_
