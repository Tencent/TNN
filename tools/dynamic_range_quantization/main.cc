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

#include "dynamic_range_quantization.h"
#include "flags.h"
#include "tnn/interpreter/tnn/model_packer.h"

using namespace TNN_NS;

void ShowUsage() {
    printf(
        "usage:\n./dynamic_range_quantization [-h] [-p] <tnnproto> [-m] <tnnmodel> [-qp] <quant_tnnproto> [-qm] <quant_tnnmodel> \n");
    printf("\t-h, <help>     \t\t\t%s\n", TNN_NS::help_message);
    printf("\t-p, <proto>    \t\t\t%s\n", TNN_NS::proto_message);
    printf("\t-m, <model>    \t\t\t%s\n", TNN_NS::model_message);
    printf("\t-qp, <quant_proto>   \t%s\n", TNN_NS::quant_proto_message);
    printf("\t-qm, <quant_model>    \t%s\n", TNN_NS::quant_model_message);
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        ShowUsage();
        return false;
    }

    if (FLAGS_p.empty() || FLAGS_m.empty() || FLAGS_qp.empty() || FLAGS_qm.empty()) {
        printf("Parameter -p/-m/-qp/-qm is not set \n");
        ShowUsage();
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return -1;
    }

    std::string tnn_proto = FLAGS_p;
    std::string tnn_model = FLAGS_m;

    std::vector<std::string> params;
    {
        std::ifstream proto_stream(tnn_proto);
        if (!proto_stream.is_open() || !proto_stream.good()) {
            LOGE("Dynamic_range_quantization: open %s failed\n", tnn_proto.c_str());
            return -1;
        }
        auto buffer = std::string((std::istreambuf_iterator<char>(proto_stream)), std::istreambuf_iterator<char>());
        params.push_back(buffer);
    }

    {
        std::ifstream model_stream(tnn_model);
        if (!model_stream.is_open() || !model_stream.good()) {
            LOGE("Dynamic_range_quantization: open %s failed\n", tnn_model.c_str());
            return -1;
        }
        auto buffer = std::string((std::istreambuf_iterator<char>(model_stream)), std::istreambuf_iterator<char>());
        params.push_back(buffer);
    }

    auto interpreter = dynamic_cast<DefaultModelInterpreter*>(CreateModelInterpreter(MODEL_TYPE_TNN));

    auto status = interpreter->Interpret(params);

    std::shared_ptr<NetStructure> net_structure(interpreter->GetNetStructure());
    std::shared_ptr<NetResource> net_resource(interpreter->GetNetResource());
    std::shared_ptr<NetStructure> quant_structure = nullptr;
    std::shared_ptr<NetResource> quant_resource   = nullptr;

    auto dynamic_range_quanter = DynamicRangeQuantizer(net_structure, net_resource);
    dynamic_range_quanter.GetDynamicRangeQuantModel(quant_structure, quant_resource);

    auto packer = std::make_shared<ModelPacker>(quant_structure.get(), quant_resource.get());

    std::string quant_tnn_proto = FLAGS_qp;
    std::string quant_tnn_model = FLAGS_qm;

    packer->Pack(quant_tnn_proto, quant_tnn_model);

    return 0;
}
