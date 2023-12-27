#include <gtest/gtest.h>

#include "test/unit_test/grad_test/grad_test.h"
#include "test/unit_test/grad_test/test_tool.h"

#include "tnn/device/arm/acc/Float4.h"

namespace TNN_NS {

TEST(InnerProductGrad, Normal) {
    GradTest grad_test;

    Input x("x", {2, 3}, {1, 2, 3, 4, 5, 6}, DATA_TYPE_FLOAT);
    Input w("w", {3, 3}, {7, 10, 13, 8, 11, 14, 9, 12, 15}, DATA_TYPE_FLOAT);  // [oc, ic]
    Input b("b", {3}, {17, 18, 19}, DATA_TYPE_FLOAT);
    std::vector<float> z_expect      = {83, 90, 97, 173, 189, 205};
    std::vector<float> b_grad_expect = {0.333333, 0.333333, 0.333333};
    std::vector<float> w_grad_expect = {0.833333, 1.166667, 1.5, 0.833333, 1.166667, 1.5, 0.833333, 1.166667, 1.5};  // [oc, ic]
    std::vector<float> x_grad_expect = {4, 5.5, 7, 4, 5.5, 7};

    std::shared_ptr<InnerProductLayerParam> layer_param = std::make_shared<InnerProductLayerParam>();
    layer_param->num_output                             = 3;
    layer_param->axis                                   = 1;
    layer_param->has_bias                               = true;

    std::shared_ptr<InnerProductLayerResource> layer_resource = std::make_shared<InnerProductLayerResource>();
    layer_resource->weight_handle = RawBuffer(w.DataBytes(), (char *)(w.data.get()), w.shape, w.dtype);
    layer_resource->bias_handle   = RawBuffer(b.DataBytes(), (char *)(b.data.get()), b.shape, b.dtype);

    std::map<std::string, std::shared_ptr<Mat>> outputs;
    // 普通输入 + 内部参数
    {
        // 训练更新w
        outputs.clear();
        Status status =
            grad_test.DoLayerGradTest(outputs, "InnerProduct", {x}, "z", layer_param, layer_resource,
                                      {"x", "z", "loss", "x_tnn_grad", "test_layer_tnn_resource_grad_0",
                                       "test_layer_tnn_resource_grad_1", "z_tnn_grad"});
        EXPECT_TRUE(status == TNN_OK);

        PrintMat(outputs);

        // 检查中间的梯度
        float *w_grad = reinterpret_cast<float *>(outputs["test_layer_tnn_resource_grad_0"]->GetData());
        float *b_grad = reinterpret_cast<float *>(outputs["test_layer_tnn_resource_grad_1"]->GetData());
        float *x_grad = reinterpret_cast<float *>(outputs["x_tnn_grad"]->GetData());
        EXPECT_TRUE(Equal(z_expect.data(), reinterpret_cast<float *>(outputs["z"]->GetData()), z_expect.size()));
        EXPECT_TRUE(Equal(x_grad_expect.data(), x_grad, x_grad_expect.size()));
        EXPECT_TRUE(Equal(w_grad_expect.data(), w_grad, w_grad_expect.size()));
        EXPECT_TRUE(Equal(b_grad_expect.data(), b_grad, b_grad_expect.size()));

        // 检查w知否做了更新
        std::vector<float> new_w(w.Size());
        float *init_w = reinterpret_cast<float *>(w.data.get());
        for (size_t i = 0; i < new_w.size(); i++) {
            new_w[i] = init_w[i] - grad_test.GetLearningRate() * w_grad[i];
        }
        EXPECT_TRUE(Equal(new_w.data(), layer_resource->weight_handle.force_to<float *>(), w.Size()));
    }
}

}  // namespace TNN_NS