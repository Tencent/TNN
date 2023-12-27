#include <gtest/gtest.h>

#include "test/unit_test/grad_test/grad_test.h"
#include "test/unit_test/grad_test/test_tool.h"

namespace TNN_NS {

// 输入维度相同，不用BroadCast
TEST(BinaryGrad, BroadCastNormal) {
    GradTest grad_test;

    std::shared_ptr<MultidirBroadcastLayerParam> layer_param = std::make_shared<MultidirBroadcastLayerParam>();
    std::shared_ptr<EltwiseLayerResource> layer_resource     = std::make_shared<EltwiseLayerResource>();

    Input x("x", {2, 3}, {1, 2, 3, 4, 5, 6}, DATA_TYPE_FLOAT);
    Input y("y", {2, 3}, {7, 8, 9, 10, 11, 12}, DATA_TYPE_FLOAT);
    std::vector<float> z      = {7, 16, 27, 40, 55, 72};
    std::vector<float> z_grad = {0.166667, 0.166667, 0.166667, 0.166667, 0.166667, 0.166667};
    std::vector<float> y_grad = {0.166667, 0.333333, 0.5, 0.666667, 0.833333, 1.0};
    std::vector<float> x_grad = {1.166667, 1.333333, 1.5, 1.666667, 1.833333, 2.0};

    std::map<std::string, std::shared_ptr<Mat>> outputs;

    // 普通输入 + 普通输入
    {
        layer_param->weight_input_index = -1;
        x.is_const = y.is_const = false;
        
        outputs.clear();
        Status status = grad_test.DoLayerGradTest(outputs, "Mul", {x, y}, "z", layer_param, nullptr,
                                                  {"x", "y", "z", "loss", "z_tnn_grad", "x_tnn_grad", "y_tnn_grad"});
        EXPECT_TRUE(status == TNN_OK);
        EXPECT_TRUE(Equal(z.data(), reinterpret_cast<float *>(outputs["z"]->GetData()), z.size()));
        EXPECT_TRUE(Equal(z_grad.data(), reinterpret_cast<float *>(outputs["z_tnn_grad"]->GetData()), z_grad.size()));
        EXPECT_TRUE(Equal(y_grad.data(), reinterpret_cast<float *>(outputs["y_tnn_grad"]->GetData()), y_grad.size()));
        EXPECT_TRUE(Equal(x_grad.data(), reinterpret_cast<float *>(outputs["x_tnn_grad"]->GetData()), x_grad.size()));
    }

    // 普通输入 + 内部参数 (w使用y的值进行测试)
    {
        layer_param->weight_input_index = 1;
        x.is_const = false;
        
        // 设置w初始值 (初始值用y的值)
        float* init_w = reinterpret_cast<float *>(y.data.get());
        auto& w = layer_resource->element_handle;
        w = RawBuffer(y.DataBytes(), (char *)init_w, y.shape, y.dtype);

        // 训练更新w
        outputs.clear();
        Status status = grad_test.DoLayerGradTest(
            outputs, "Mul", {x}, "z", layer_param, layer_resource,
            {"x", "z", "loss", "z_tnn_grad", "x_tnn_grad", "test_layer_tnn_resource_grad_0", "z_tnn_grad"});
        EXPECT_TRUE(status == TNN_OK);

        // 检查中间的梯度
        float *w_grad = reinterpret_cast<float *>(outputs["test_layer_tnn_resource_grad_0"]->GetData());
        EXPECT_TRUE(Equal(z.data(), reinterpret_cast<float *>(outputs["z"]->GetData()), z.size()));
        EXPECT_TRUE(Equal(z_grad.data(), reinterpret_cast<float *>(outputs["z_tnn_grad"]->GetData()), z_grad.size()));
        EXPECT_TRUE(Equal(y_grad.data(), w_grad, y_grad.size()));
        EXPECT_TRUE(Equal(x_grad.data(), reinterpret_cast<float *>(outputs["x_tnn_grad"]->GetData()), x_grad.size()));

        // 检查w知否做了更新
        std::vector<float> new_w(w.GetDataCount());
        for (size_t i = 0; i < new_w.size(); i++) {
            new_w[i] = init_w[i] - grad_test.GetLearningRate() * w_grad[i];
        }
        EXPECT_TRUE(Equal(new_w.data(), w.force_to<float *>(), w.GetDataCount()));
    }

    // 内部参数（w使用x的值进行测试） + 普通输入
    {
        layer_param->weight_input_index = 0;
        y.is_const = false;
        
        // 设置w初始值 (初始值用y的值)
        float* init_w = reinterpret_cast<float *>(x.data.get());
        auto& w = layer_resource->element_handle;
        w = RawBuffer(x.DataBytes(), (char *)init_w, x.shape, x.dtype);

        // 训练更新w
        outputs.clear();
        Status status = grad_test.DoLayerGradTest(
            outputs, "Mul", {y}, "z", layer_param, layer_resource,
            {"y", "z", "loss", "z_tnn_grad", "y_tnn_grad", "test_layer_tnn_resource_grad_0", "z_tnn_grad"});
        EXPECT_TRUE(status == TNN_OK);
        PrintMat(outputs);

        // 检查中间的梯度
        float *w_grad = reinterpret_cast<float *>(outputs["test_layer_tnn_resource_grad_0"]->GetData());
        EXPECT_TRUE(Equal(z.data(), reinterpret_cast<float *>(outputs["z"]->GetData()), z.size()));
        EXPECT_TRUE(Equal(z_grad.data(), reinterpret_cast<float *>(outputs["z_tnn_grad"]->GetData()), z_grad.size()));
        EXPECT_TRUE(Equal(x_grad.data(), w_grad, x_grad.size()));
        EXPECT_TRUE(Equal(y_grad.data(), reinterpret_cast<float *>(outputs["y_tnn_grad"]->GetData()), y_grad.size()));

        // 检查w知否做了更新
        std::vector<float> new_w(w.GetDataCount());
        for (size_t i = 0; i < new_w.size(); i++) {
            new_w[i] = init_w[i] - grad_test.GetLearningRate() * w_grad[i];
        }
        EXPECT_TRUE(Equal(new_w.data(), w.force_to<float *>(), w.GetDataCount()));
    }

    // 常量输入 + 普通输入
    {
        layer_param->weight_input_index = -1;
        x.is_const = true;
        y.is_const = false;
        
        outputs.clear();
        Status status = grad_test.DoLayerGradTest(outputs, "Mul", {x, y}, "z", layer_param, nullptr,
                                                  {"x", "y", "z", "loss", "z_tnn_grad", "x_tnn_grad", "y_tnn_grad"});
        EXPECT_TRUE(status == TNN_OK);
        EXPECT_TRUE(Equal(z.data(), reinterpret_cast<float *>(outputs["z"]->GetData()), z.size()));
        EXPECT_TRUE(Equal(z_grad.data(), reinterpret_cast<float *>(outputs["z_tnn_grad"]->GetData()), z_grad.size()));
        EXPECT_TRUE(Equal(y_grad.data(), reinterpret_cast<float *>(outputs["y_tnn_grad"]->GetData()), y_grad.size()));
        EXPECT_TRUE(Equal(x_grad.data(), reinterpret_cast<float *>(outputs["x_tnn_grad"]->GetData()), x_grad.size()));
    }
}

// 输入维度不同，BroadCastSingle
TEST(BinaryGrad, BroadCastSingle) {
    GradTest grad_test;

    std::shared_ptr<MultidirBroadcastLayerParam> layer_param = std::make_shared<MultidirBroadcastLayerParam>();
    std::shared_ptr<EltwiseLayerResource> layer_resource     = std::make_shared<EltwiseLayerResource>();

    Input x("x", {1}, {5}, DATA_TYPE_FLOAT);
    Input y("y", {2, 5}, {6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, DATA_TYPE_FLOAT);
    std::vector<float> z      = {30, 35, 40, 45, 50, 55, 60, 65, 70, 75};
    std::vector<float> z_grad = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    std::vector<float> y_grad = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    std::vector<float> x_grad = {10.5};

    std::map<std::string, std::shared_ptr<Mat>> outputs;

    // 普通输入(需要扩展) + 普通输入（不需要扩展）
    {
        layer_param->weight_input_index = -1;
        x.is_const = y.is_const = false;
        
        outputs.clear();
        Status status = grad_test.DoLayerGradTest(outputs, "Mul", {x, y}, "z", layer_param, nullptr,
                                                  {"x", "y", "z", "loss", "z_tnn_grad", "x_tnn_grad", "y_tnn_grad"});
        EXPECT_TRUE(status == TNN_OK);
        EXPECT_TRUE(Equal(z.data(), reinterpret_cast<float *>(outputs["z"]->GetData()), z.size()));
        EXPECT_TRUE(Equal(z_grad.data(), reinterpret_cast<float *>(outputs["z_tnn_grad"]->GetData()), z_grad.size()));
        EXPECT_TRUE(Equal(y_grad.data(), reinterpret_cast<float *>(outputs["y_tnn_grad"]->GetData()), y_grad.size()));
        EXPECT_TRUE(Equal(x_grad.data(), reinterpret_cast<float *>(outputs["x_tnn_grad"]->GetData()), x_grad.size()));
    }
}

}  // namespace TNN_NS