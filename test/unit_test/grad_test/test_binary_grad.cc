#include <gtest/gtest.h>

#include "test/unit_test/grad_test/grad_test.h"
#include "test/unit_test/grad_test/test_tool.h"

namespace TNN_NS {

// 输入维度相同，不用BroadCast
TEST(BinaryGrad, NoBroadCast) {
    GradTest grad_test;

    std::shared_ptr<MultidirBroadcastLayerParam> layer_param = std::make_shared<MultidirBroadcastLayerParam>();
    std::shared_ptr<EltwiseLayerResource> layer_resource     = std::make_shared<EltwiseLayerResource>();
    Input x("x", {2, 3}, {1, 2, 3, 4, 5, 6}, DATA_TYPE_FLOAT);
    Input y("y", {2, 3}, {7, 8, 9, 10, 11, 12}, DATA_TYPE_FLOAT);
    std::vector<float> z      = {7, 16, 27, 40, 55, 72};
    std::vector<float> z_grad = {0.166667, 0.166667, 0.166667, 0.166667, 0.166667, 0.166667};
    std::vector<float> y_grad = {0.166667, 0.333333, 0.500000, 0.666667, 0.833333, 1.000000};
    std::vector<float> x_grad = {1.166667, 1.333333, 1.500000, 1.666667, 1.833333, 2.000000};

    // 普通输入 + 普通输入
    {
        layer_param->weight_input_index = -1;
        x.is_const = y.is_const = false;
        auto outputs            = grad_test.DoLayerGradTest("Mul", {x, y}, "z", layer_param, nullptr,
                                                            {"x", "y", "z", "loss", "z_tnn_grad", "x_tnn_grad", "y_tnn_grad"});
        PrintMat(outputs);
        EXPECT_TRUE(Equal(z.data(), reinterpret_cast<float *>(outputs["z"]->GetData()), z.size()));
        EXPECT_TRUE(Equal(z_grad.data(), reinterpret_cast<float *>(outputs["z_tnn_grad"]->GetData()), z_grad.size()));
        EXPECT_TRUE(Equal(y_grad.data(), reinterpret_cast<float *>(outputs["y_tnn_grad"]->GetData()), y_grad.size()));
        EXPECT_TRUE(Equal(x_grad.data(), reinterpret_cast<float *>(outputs["x_tnn_grad"]->GetData()), x_grad.size()));
    }

    // 普通输入 + 内部参数
    layer_param->weight_input_index = 1;

    // 内部参数 + 普通输入
    layer_param->weight_input_index = 0;

    // 常量输入 + 普通输入
    layer_param->weight_input_index = -1;

    // 普通输入 + 常量输入
    layer_param->weight_input_index = -1;

    // 常量输入 + 常量输入
    layer_param->weight_input_index = -1;
}

}  // namespace TNN_NS