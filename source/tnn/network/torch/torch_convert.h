#pragma once

#include <vector>

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
#include "tnn/network/torch/torch_tnn_runtime.h"

namespace TNN_NS {
namespace conversion {

c10::intrusive_ptr<runtime::TNNEngine> ConvertBlockToInstance(partitioning::SegmentedBlock &block, NetworkConfig &config); 

}
}