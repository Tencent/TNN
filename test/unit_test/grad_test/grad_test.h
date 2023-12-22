#pragma once

#include "test/unit_test/grad_test/test_tool.h"

namespace TNN_NS {

class GradTest {
public:
    // 测试一个Layer的反向计算，把想关注的Blob放到net_outputs里，作为Mat输出出来
    std::map<std::string, std::shared_ptr<Mat>> DoLayerGradTest(const std::string& layer_type,
                                                                const std::vector<Input>& layer_inputs,
                                                                const std::string& layer_output,
                                                                std::shared_ptr<LayerParam> layer_param,
                                                                std::shared_ptr<LayerResource> layer_resource,
                                                                const std::set<std::string>& net_outputs);
private:
    // 创建一个Layer的后向测试网络：包含两个layer，一个是要测试的layer，一个是ReduceMean用来产生一个标量作为loss
    // 1) 学习率：0.1
    // 2) Loss使用ReduceMean产生，则要测试的Layer的 dL/dy = 1 / count(y.dims)
    std::shared_ptr<Instance> InitLayerGradTestNetwork(
        const std::string& layer_type, const std::vector<Input>& layer_inputs,
        const std::vector<std::string>& layer_outputs, std::shared_ptr<LayerParam> layer_param,
        const std::set<std::string>& net_outputs, std::shared_ptr<LayerResource> layer_resource = nullptr,
        const std::map<std::string, std::shared_ptr<RawBuffer>>& consts = {});

    NetworkConfig GetNetworkConfig();
};

}  // namespace TNN_NS
