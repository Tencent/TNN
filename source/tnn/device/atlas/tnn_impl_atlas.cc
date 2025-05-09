// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <fstream>
#include "tnn/core/instance.h"
#include "tnn/device/atlas/atlas_network.h"
#include "tnn/device/atlas/atlas_utils.h"
#include "tnn/device/atlas/tnn_impl_atlas.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

TNNImplFactoryRegister<TNNImplFactory<TNNImplAtlas>> g_tnn_impl_atlas_factory_register(MODEL_TYPE_ATLAS);

TNNImplAtlas::TNNImplAtlas() {}

TNNImplAtlas::~TNNImplAtlas() {}

Status TNNImplAtlas::Init(ModelConfig& config) {
    TNNImpl::Init(config);
    
    this->model_type_ = config.model_type;

    if (config.model_type == TNN_NS::MODEL_TYPE_TNN ||
        config.model_type == TNN_NS::MODEL_TYPE_RAPIDNET ||
        config.model_type == TNN_NS::MODEL_TYPE_ATLAS) {
        LOGD("Model Type is TNN or ATLAS OM, ACL API Required. Call aclInit() ...\n");
        aclError acl_ret = aclInit(nullptr);
        if (acl_ret != ACL_ERROR_NONE && acl_ret != ACL_ERROR_REPEAT_INITIALIZE) {
            LOGE("Atlas API: aclInit failed!\n");
            return TNNERR_ATLAS_RUNTIME_ERROR;
        }
        LOGD("Model Type is TNN or ATLAS OM, ACL API Required. Call aclInit() ... done.\n");
        this->acl_init_called_ = true;
    }

    auto interpreter = CreateModelInterpreter(config.model_type);
    if (!interpreter) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }
    interpreter_ = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    return interpreter_->Interpret(config.params);
}

Status TNNImplAtlas::DeInit() {
    if (this->acl_init_called_) {
        LOGD("TNNImplAtlas DeInit: to call aclFinalize().\n");
        aclError ret = aclFinalize();
        if (ret != ACL_ERROR_NONE) {
            LOGD("TNNImplAtlas DeInit: ATLAS API: aclFinalize failed!\n");
        }
    }

    return TNN_OK;
}

Status TNNImplAtlas::AddOutput(const std::string& layer_name, int output_index) {
    LOGE("AddOutput() API not supported on TNN ATLAS.\n");
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "AddOutput() API not supported on TNN ATLAS.\n");
}

Status TNNImplAtlas::GetModelInputNames(std::vector<std::string>& input_names) {
    if (model_type_ == MODEL_TYPE_ATLAS) {
        if (this->om_model_desc_of_the_first_instance_ == nullptr) {
            LOGE("Fail to Get TNN Atlas ModelInputNames, model desc missing.");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to Get TNN Atlas ModelInputNames, model desc missing.");
        }
        
        size_t num_inputs = aclmdlGetNumInputs(this->om_model_desc_of_the_first_instance_);
        std::vector<std::string> in_names_vec;
        for (size_t i=0; i<num_inputs; i++) {
            std::string input_name;
            input_name.assign(aclmdlGetInputNameByIndex(this->om_model_desc_of_the_first_instance_, i));
            in_names_vec.emplace_back(input_name);
        }
        input_names = in_names_vec;
    } else {
        LOGE("API not supported for current MODEL TYPE.\n");
    }

    return TNN_OK;
}

Status TNNImplAtlas::GetModelOutputNames(std::vector<std::string>& output_names) {
    if (model_type_ == MODEL_TYPE_ATLAS) {
        if (this->om_model_desc_of_the_first_instance_ == nullptr) {
            LOGE("Fail to Get TNN Atlas ModelOutputNames, model desc missing.\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to Get TNN Atlas ModelOutputNames, model desc missing.");
        }
        
        size_t num_outputs = aclmdlGetNumOutputs(this->om_model_desc_of_the_first_instance_);
        std::vector<std::string> out_names_vec;
        for (size_t i=0; i<num_outputs; i++) {
            std::string output_name;
            output_name.assign(aclmdlGetOutputNameByIndex(this->om_model_desc_of_the_first_instance_, i));
            out_names_vec.emplace_back(output_name);
        }
        output_names = out_names_vec;
    } else {
        LOGE("API not supported for current MODEL TYPE.\n");
    }

    return TNN_OK;
}

Status TNNImplAtlas::GetModelInputShapesMap(InputShapesMap& shapes_map) {
    if (model_type_ == MODEL_TYPE_ATLAS) {
        if (this->om_model_desc_of_the_first_instance_ == nullptr) {
            LOGE("Fail to Get TNN Atlas ModelInputNames, model desc missing.\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to Get TNN Atlas ModelInputNames, model desc missing.");
        }
        
        size_t num_inputs = aclmdlGetNumInputs(this->om_model_desc_of_the_first_instance_);
        InputShapesMap in_shapes_map;
        for (size_t i=0; i<num_inputs; i++) {
            aclmdlIODims acl_dims;
            aclError acl_ret = aclmdlGetInputDims(this->om_model_desc_of_the_first_instance_, i, &acl_dims);
            if (acl_ret != ACL_ERROR_NONE) {
                LOGE("acl get input dim failed (acl error code: %d)\n", acl_ret);
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl get input dim falied");
            }
            std::string input_name;
            input_name.assign(aclmdlGetInputNameByIndex(this->om_model_desc_of_the_first_instance_, i));
            std::vector<int> in_dims;
            for (int d=0; d<std::min(int(acl_dims.dimCount),7); d++) { // Max Dim Allowed is 6.
                if (acl_dims.dims[d]!=0) {
                    in_dims.push_back(acl_dims.dims[d]);
                } else {
                    break;
                }
            }
            in_shapes_map[input_name] = in_dims;
        }
        shapes_map = in_shapes_map;
    } else {
        LOGE("API not supported for current MODEL TYPE.\n");
    }
    return TNN_OK;
}

Status TNNImplAtlas::GetModelInputDataTypeMap(InputDataTypeMap& data_type_map) {
    if (model_type_ == MODEL_TYPE_ATLAS) {
        if (this->om_model_desc_of_the_first_instance_ == nullptr) {
            LOGE("Fail to Get TNN Atlas ModelInputNames, model desc missing.\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to Get TNN Atlas ModelInputNames, model desc missing.");
        }
        
        size_t num_inputs = aclmdlGetNumInputs(this->om_model_desc_of_the_first_instance_);
        InputDataTypeMap in_dtype_map;
        for (size_t i=0; i<num_inputs; i++) {
            std::string input_name;
            input_name.assign(aclmdlGetInputNameByIndex(this->om_model_desc_of_the_first_instance_, i));
            aclDataType acl_dtype = aclmdlGetInputDataType(this->om_model_desc_of_the_first_instance_, i);
            DataType tnn_dtype;
            aclError acl_ret = ConvertFromAclDataTypeToTnnDataType(acl_dtype, tnn_dtype);
            if (acl_ret != ACL_ERROR_NONE) {
                LOGE("acl get input data type failed, maybe unsupported data type (acl error code: %d)\n", acl_ret);
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl get input data type failed");
            }
            in_dtype_map[input_name] = tnn_dtype;
        }
        data_type_map = in_dtype_map;
    } else {
        LOGE("API not supported for current MODEL TYPE.\n");
    }
    return TNN_OK;
}

std::shared_ptr<Instance> TNNImplAtlas::CreateInst(NetworkConfig& net_config, Status& status,
                                                   InputShapesMap inputs_shape, InputDataTypeMap inputs_data_type) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, inputs_shape, inputs_data_type);
    return instance;
}

std::shared_ptr<Instance> TNNImplAtlas::CreateInst(NetworkConfig& net_config, Status& status,
                                                   InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, 
                                                   InputDataTypeMap inputs_data_type) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, min_inputs_shape, max_inputs_shape, inputs_data_type);

    if (model_type_ == MODEL_TYPE_ATLAS) {
        AtlasNetwork* atlas_net = reinterpret_cast<AtlasNetwork*>(instance->GetNetwork());
        if (this->om_model_id_of_the_first_instance_ == 0) {
            this->om_model_id_of_the_first_instance_ = atlas_net->GetOMModelInfo()->model_id;
            LOGD("TNNImplAtlas init the first Instance, get model id.\n");
        }
        if (this->om_model_desc_of_the_first_instance_ == nullptr) {
            this->om_model_desc_of_the_first_instance_ = atlas_net->GetOMModelInfo()->model_desc;
            LOGD("TNNImplAtlas init the first Instance, get model desc.\n");
        }
    }

    return instance;
}

}  // namespace TNN_NS
