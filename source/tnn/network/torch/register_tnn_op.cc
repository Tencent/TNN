
#include "tnn/core/blob.h"
#include "tnn/network/torch/jit_util.h"
#include "tnn/network/torch/torch_convert.h"
#include "tnn/network/torch/torch_tnn_runtime.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/torch.h"

// #include "/root/env/pytorch-1.8.1/lib/python3.7/site-packages/torch/include/ATen/core/jit_type.h"

namespace TNN_NS {
namespace runtime {

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs,
                                        c10::intrusive_ptr<TNNEngine> compiled_engine) {
    auto input_names = compiled_engine->input_names;
    InputShapesMap inputs_shape_map;
    int input_idx = 0;
    for (auto &input : inputs) {
        inputs_shape_map[input_names[input_idx++]] = util::toDims(input.sizes());
    }

    if (!compiled_engine->is_init_) {
        auto interpreter = dynamic_cast<DefaultModelInterpreter *>(compiled_engine->ctx_->get_interpreter().get());
        interpreter->GetNetStructure()->inputs_shape_map = inputs_shape_map;
        compiled_engine->instance_->Init(compiled_engine->ctx_->get_interpreter(), inputs_shape_map);
        compiled_engine->is_init_ = true;
    }

    BlobMap input_blobs;
    BlobMap output_blobs;
    compiled_engine->instance_->GetAllInputBlobs(input_blobs);
    compiled_engine->instance_->GetAllOutputBlobs(output_blobs);

    // TRTORCH_CHECK(
    //     compiled_engine->exec_ctx->allInputDimensionsSpecified(), "Not enough inputs provided
    //     (runtime.RunCudaEngine)");
    compiled_engine->instance_->Forward();
    std::cout << "tnn engine work!!!" << std::endl;

    std::vector<at::Tensor> outputs(output_blobs.size());

    auto tnn_dims = output_blobs.begin()->second->GetBlobDesc().dims;
    std::vector<int64_t> dims;
    for (auto iter : tnn_dims)
        dims.push_back(iter);
    auto type  = at::kFloat;
    outputs[0] = std::move(at::empty(dims, {at::kCUDA}).to(type).contiguous());

    return outputs;
}

static auto TNNEngineTSRegistrtion = torch::class_<TNNEngine>("tnn", "Engine");

TORCH_LIBRARY(tnn, m) {
    // auto type_ptr = c10::detail::getTypePtr_<c10::intrusive_ptr<TNNEngine>>::call();
    m.def("execute_engine", execute_engine);
}

}  // namespace runtime
}  // namespace TNN_NS
