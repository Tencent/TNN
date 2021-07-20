#include "tnn/network/torch/torch_tnn_runtime.h"
#include "torch/custom_class.h"

namespace TNN_NS {
namespace runtime {

TNNEngine::TNNEngine(NetworkConfig& network_config, ModelConfig& model_config) {
    instance_ = std::make_shared<Instance>(network_config, model_config);
}
TNNEngine::TNNEngine(std::shared_ptr<Instance> &instance) {
    instance_ = instance;
}

TNNEngine& TNNEngine::operator=(const TNNEngine& other) {
    instance_ = other.instance_;
    return (*this);
}

}  // namespace runtime
}  // namespace TNN_NS