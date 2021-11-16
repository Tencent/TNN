#pragma once

#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/core/abstract_network.h"
#include "tnn/core/default_network.h"
#include "tnn/core/blob.h"
#include "tnn/core/blob_manager.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/layer/base_layer.h"

#include "tnn/network/torch/segment.h"
#include "tnn/interpreter/default_model_interpreter.h"

#include <torch/script.h>
#include "c10/util/intrusive_ptr.h"
#include "torch/custom_class.h"
namespace TNN_NS{
namespace runtime {

const std::string TORCH_DELIM     = "%";
const std::string TORCH_INT_DELIM = ",";

typedef enum { 
    PROTO_IDX = 0,
    MODEL_IDX,
    INPUT_NAME_IDX,
    OUTPUT_NAME_IDX,
    MIN_SHAPE_IDX,
    MAX_SHAPE_IDX,
    CONFIG_IDX,
    CACHE_IDX,
    ENDING_IDX
} SerializedInfoIndex;

template <typename T>
inline std::string Serialize(std::vector<T> contents, const std::string delim = TORCH_DELIM) {
    std::stringstream ss;
    for (size_t i = 0; i < contents.size() - 1; i++) {
        ss << contents[i] << delim;
    }
    ss << contents[contents.size() - 1];
    return ss.str(); 
}

inline std::vector<std::string> Deserialize(std::string content, const std::string delim = TORCH_DELIM) {
    std::vector<std::string> tokens;
    int64_t start = 0;
    int64_t end = content.find(delim);

    while (end != -1) {
        tokens.push_back(content.substr(start, end - start));
        start = end + delim.size();
        end = content.find(delim, start);
    }
    tokens.push_back(content.substr(start, end - start));

    return tokens;
}

class TorchConvertCtx {
public:
    TorchConvertCtx() {
        auto interpreter = CreateModelInterpreter(MODEL_TYPE_TNNIR);
        interpreter_ = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    };
    void buildOp(const torch::jit::Node* node);
    bool dealPrime(const torch::jit::Node* node);
    int declareTensor(std::string name);
    int lookupTensor(std::string name);
    std::string lookupTensor(int idx) const;
    void declareVar(std::string name, const torch::jit::Node* var);
    const torch::jit::Node* lookupVar(std::string name) const;
    std::shared_ptr<AbstractModelInterpreter> get_interpreter() {
        return interpreter_;
    };
    void set_interpreter(std::shared_ptr<AbstractModelInterpreter> interpreter) {
        interpreter_ = interpreter;
    };
private:
    std::map<std::string, const torch::jit::Node*> varTable;
    std::map<std::string, int> tensorIdx;
    std::shared_ptr<AbstractModelInterpreter> interpreter_;
};

struct TNNEngine : torch::CustomClassHolder {
    TNNEngine(NetworkConfig &network_config, ModelConfig &model_config);
    TNNEngine(std::vector<std::string> &serialize);
    std::shared_ptr<Instance> instance_;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<DimsVector> min_inputs_shape;
    std::vector<DimsVector> max_inputs_shape;

    NetworkConfig network_config_;

    std::shared_ptr<TorchConvertCtx> ctx_;
    bool is_init_ = false;

    TNNEngine& operator=(const TNNEngine& other); 
};

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TNNEngine> compiled_engine);
}
}