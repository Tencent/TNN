
#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/torch.h"
#include "tnn/network/torch/torch_convert.h"
#include "tnn/network/torch/jit_util.h"
#include "tnn/core/blob.h"
#include "tnn/network/torch/torch_tnn_runtime.h"

// #include "/root/env/pytorch-1.8.1/lib/python3.7/site-packages/torch/include/ATen/core/jit_type.h"

namespace TNN_NS {
namespace runtime {

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TNNEngine> compiled_engine) {
    BlobMap input_blobs;
    BlobMap output_blobs;
    compiled_engine->instance_->GetAllInputBlobs(input_blobs);
    compiled_engine->instance_->GetAllOutputBlobs(output_blobs);
  // LOG_DEBUG("Attempting to run engine (ID: " << compiled_engine->name << ")");
  // std::vector<void*> gpu_handles;

  // std::vector<at::Tensor> contig_inputs{};
  // contig_inputs.reserve(inputs.size());

  // for (size_t i = 0; i < inputs.size(); i++) {
  //   uint64_t pyt_idx = compiled_engine->in_binding_map[i];
  //   TRTORCH_CHECK(
  //       inputs[pyt_idx].is_cuda(),
  //       "Expected input tensors to have device cuda, found device " << inputs[pyt_idx].device());
  //   auto expected_type = util::toATenDType(compiled_engine->exec_ctx->getEngine().getBindingDataType(i));
  //   TRTORCH_CHECK(
  //       inputs[pyt_idx].dtype() == expected_type,
  //       "Expected input tensors to have type " << expected_type << ", found type " << inputs[pyt_idx].dtype());
  //   auto dims = core::util::toDimsPad(inputs[pyt_idx].sizes(), 1);
  //   auto shape = core::util::toVec(dims);
  //   contig_inputs.push_back(inputs[pyt_idx].view(shape).contiguous());
  //   LOG_DEBUG("Input shape: " << dims);
  //   compiled_engine->exec_ctx->setBindingDimensions(i, dims);
  //   gpu_handles.push_back(contig_inputs.back().data_ptr());
  // }

  // TRTORCH_CHECK(
  //     compiled_engine->exec_ctx->allInputDimensionsSpecified(), "Not enough inputs provided (runtime.RunCudaEngine)");
  std::cout << "tnn engine work!!!" << std::endl;

  std::vector<at::Tensor> outputs(output_blobs.size());

  auto tnn_dims = output_blobs.begin()->second->GetBlobDesc().dims;
  std::vector<int64_t> dims;
  for (auto iter : tnn_dims) dims.push_back(iter);
  auto type = at::kFloat;
  outputs[0] = std::move(at::empty(dims, {at::kCUDA}).to(type).contiguous());

  return outputs;
}

static auto TNNEngineTSRegistrtion = 
    torch::class_<TNNEngine>("tnn", "Engine");

TORCH_LIBRARY(tnn, m) {
  // auto type_ptr = c10::detail::getTypePtr_<c10::intrusive_ptr<TNNEngine>>::call();
  m.def("execute_engine", execute_engine);
}

} // namespace runtime
} // namespace TNN_NS
