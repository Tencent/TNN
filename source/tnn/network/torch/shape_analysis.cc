#include "tnn/network/torch/shape_analysis.h"
#include "tnn/core/macro.h"
#include "tnn/network/torch/jit_util.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/passes/constant_pooling.h"

namespace TNN_NS {
namespace partitioning {

std::vector<torch::jit::IValue> generateRandomInputs(InputShapesMap& input_shape) {
  // generate random inputs for running pytorch segments
  std::vector<torch::jit::IValue> random_inputs;
  if (input_shape.size() > 1) {
    std::cout << "!!! can not support multi inputs now" << std::endl;
    return random_inputs;
  }
  for (auto& input_range : input_shape) {
    auto cur_shape = input_range.second;
    std::vector<int64_t> shape;
    shape.insert(shape.begin(), std::begin(cur_shape), std::begin(cur_shape) + cur_shape.size());
    auto in = at::randint(5, shape, {at::kCUDA});
    random_inputs.push_back(in.clone());
  }
  return random_inputs;
}

void getSegmentsOutputByRunning(
    SegmentedBlock& seg_block,
    std::unordered_map<torch::jit::Value*, torch::jit::IValue>& ivalues_maps) {
  // create a module to run the graph
  auto g = seg_block.g();
  auto copy_g = g->copy();

  // create tuple for multiple outputs
  if (seg_block.raw_outputs().size() > 1) {
    auto new_output_node = copy_g->appendNode(copy_g->createTuple(copy_g->outputs()));
    for (int idx = copy_g->outputs().size() - 1; idx >= 0; --idx) {
      copy_g->eraseOutput(idx);
    }

    copy_g->registerOutput(new_output_node->outputs()[0]);
  }

  torch::jit::script::Module cur_mod(c10::QualifiedName("module"));

  auto self = copy_g->insertInput(0, "self_1");
  self->setType(cur_mod.type());

  auto cur_method = cur_mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), copy_g);
  auto schema = util::GenerateGraphSchema(cur_method->name(), copy_g);
  cur_mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;

  // set inputs ivalues, now supports Tensor/Int to pass argumentes between different segments
  for (auto& input : seg_block.raw_inputs()) {
    printf("seg input name %s\n", input->debugName().c_str());
    if (!ivalues_maps.count(input)) {
      std::cout << "Could not find mini graph input IValue " << input->debugName() << std::endl;
      return;
    }
    if (input->node()->kind() == torch::jit::prim::Param) {
      jit_inputs_ivalues.push_back(ivalues_maps[input]);
    } else if (input->type()->isSubtypeOf(torch::jit::TensorType::get())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toTensor());
    } else if (input->type()->isSubtypeOf(torch::jit::IntType::get())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toInt());
    } else if (input->type()->isSubtypeOf(torch::jit::BoolType::get())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toBool());
    } else if (input->type()->kind() == torch::jit::TypeKind::ListType) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toList());
    } else if (input->type()->kind() == torch::jit::TypeKind::TupleType) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toTuple());
    } else {
      std::cout << "Unable to find type for value: " << input->debugName() << " to get the ivalues.\n";
      return;
    }
  }

  // run segments to get outputs for later segments input shape, and other arguments such as Int
  std::vector<torch::jit::IValue> jit_results;
  torch::jit::IValue jit_results_ivalues = cur_mod.forward(jit_inputs_ivalues);

  if (jit_results_ivalues.isTuple()) {
    auto results = jit_results_ivalues.toTuple()->elements();
    for (auto r : results) {
      jit_results.push_back(r);
    }
  } else {
    jit_results.push_back(jit_results_ivalues);
  }

  size_t idx = 0;
  for (auto& output : seg_block.raw_outputs()) {
    ivalues_maps[output] = jit_results[idx++];
  }

  // set input shape for each segmented block so we wil use it in conversion process
  std::vector<DimsVector> input_shape;
  for (auto& i : seg_block.raw_inputs()) {
    if (ivalues_maps[i].isTensor()) {
      input_shape.push_back(util::toDims(ivalues_maps[i].toTensor().sizes()));
    }
  }

  seg_block.register_inshape(input_shape);
}

void runShapeAnalysis(
    std::vector<SegmentedBlock>& segmented_blocks,
    InputShapesMap& input_shape,
    std::shared_ptr<torch::jit::Graph> g) {
  // store the mapping from lowering graph torch::jit::Value => torch::jit::IValue that we get by running segments
  std::unordered_map<torch::jit::Value*, torch::jit::IValue> ivalues_maps;
  std::vector<torch::jit::IValue> random_inputs = generateRandomInputs(input_shape);
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    printf("graph input %d %s\n", i, g->inputs()[i]->debugName().c_str());
    ivalues_maps[g->inputs()[i]] = random_inputs[i];
  }

  // register every segment's input shape, and it's running output IValues
  for (auto& seg_block : segmented_blocks) {
    torch::jit::ConstantPooling(seg_block.g());
    getSegmentsOutputByRunning(seg_block, ivalues_maps);
  }
  return;
}

} // namespace partitioning
} // namespace TNN_NS
