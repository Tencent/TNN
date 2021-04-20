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

#ifndef TNN_INCLUDE_TNN_CORE_INSTANCE_H_
#define TNN_INCLUDE_TNN_CORE_INSTANCE_H_

#include <functional>
#include <memory>
#include <vector>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/utils/blob_converter.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace TNN_NS {

class AbstractNetwork;
class AbstractModelInterpreter;

struct LayerInfo;

#ifdef FORWARD_CALLBACK_ENABLE
typedef std::function<void(std::vector<Blob*>& blobs, LayerInfo* info)> BlobStatisticCallback;
#endif  // end of FORWARD_CALLBACK_ENABLE

class PUBLIC Instance {
public:
    Instance(NetworkConfig& net_config, ModelConfig& model_config);

    ~Instance();

    // init with model interpeter and inputs shape.
    Status Init(std::shared_ptr<AbstractModelInterpreter> interpreter, InputShapesMap inputs_shape);

    // init with model interpeter, min inputs shape and max inputs shape.
    Status Init(std::shared_ptr<AbstractModelInterpreter> interpreter, InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape);

    // deinit, release network
    Status DeInit();

    //  return memory bytes required for forward
    Status GetForwardMemorySize(int& memory_size);

    //  set memory to tnn instance. if success, return status code zero.
    //  only instance created with SHARE_MEMORY_MODE_SET_FROM_EXTERNAL can be set from external.
    //  the memory size need >=  GetForwardMemorySize().
    //  releasing or otherwise using the memory for other purposes during the tnn network run
    //  will result in undefined behavior.
    Status SetForwardMemory(void* memory);

    // reshape instance with new input shapes
    Status Reshape(const InputShapesMap& inputs);

    // get tnn command queue
    Status GetCommandQueue(void** command_queue);
    
    // @brief share command queue with another instance
    // @param instance to share command queue
    Status ShareCommandQueue(Instance *instance);

    // @brief tnn instance network infer, it will wait until all layer infer complete.
    Status Forward();

#ifdef FORWARD_CALLBACK_ENABLE
    // tnn instance network infer with callback to get blob info
    Status ForwardWithCallback(BlobStatisticCallback before, BlobStatisticCallback after);
#endif  // end of FORWARD_CALLBACK_ENABLE

#ifdef GET_INTERP_ENABLE
    // get model interpreter
    std::shared_ptr<AbstractModelInterpreter> GetInterpreter();
#endif  // end of GET_INTERP_ENABLE

    // tnn instance network infer async.
    // device gpu, all layer infer complete will call Callback.
    Status ForwardAsync(Callback call_back);

    // get all input blobs
    Status GetAllInputBlobs(BlobMap& blobs);

    // get all output blobs
    Status GetAllOutputBlobs(BlobMap& blobs);

    // set threads run on cpu
    Status SetCpuNumThreads(int num_threads);

#if TNN_PROFILE
public:
    /**start to profile each layer, dont call this func if you only want to profile the whole mode*/
    void StartProfile();
    /**finish profile each layer and show result*/
    std::string FinishProfile(bool do_print = false);
#endif

private:
    std::shared_ptr<AbstractModelInterpreter> interpreter_ = nullptr;
    std::shared_ptr<AbstractNetwork> network_ = nullptr;
    std::shared_ptr<AbstractNetwork> const_folder_ = nullptr;
    NetworkConfig net_config_;
    ModelConfig model_config_;
    
    AbstractNetwork *GetNetwork();
    
    //Mat interface for simple use
public:
    // set input Mat, if input_name is not set, take the first input as default
    Status SetInputMat(std::shared_ptr<Mat> mat,
                       MatConvertParam param,
                       std::string input_name = "");
    
    // get output Mat, if output_name is not set, take the first output as default
    Status GetOutputMat(std::shared_ptr<Mat>& mat,
                        MatConvertParam param = MatConvertParam(),
                        std::string output_name = "",
                        DeviceType device = DEVICE_ARM, MatType mat_type = NCHW_FLOAT);
    
private:
    // input converter
    std::map<std::string, std::shared_ptr<BlobConverter>> input_converters_ = {};

    // output converter
    std::map<std::string, std::shared_ptr<BlobConverter>> output_converters_ = {};

    // output mat
    std::map<std::string, std::shared_ptr<Mat>> output_mats_ = {};
    // output mat convert status
    std::map<std::string, int> output_mats_convert_status_ = {};
};

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_INCLUDE_TNN_CORE_INSTANCE_H_
