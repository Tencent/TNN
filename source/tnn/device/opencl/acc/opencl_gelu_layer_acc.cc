#include "tnn/device/opencl/acc/opencl_unary_layer_acc.h"

namespace TNN_NS {

DECLARE_OPENCL_UNARY_ACC(Gelu);

Status OpenCLGeluLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Gelu ACC\n");
    Status ret = OpenCLUnaryLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret);

    op_name_ = "Gelu";

    return TNN_OK;
}

std::set<std::string> OpenCLGeluLayerAcc::CreateBuildOptions() {
    std::set<std::string> build_options;
    // std::string compute = "(FLOAT4)(0.5f)*in*(erf(in*(FLOAT4)(0.707106793288165f))+(FLOAT4)(1.f))";
    std::string compute = "(FLOAT4)(0.5f)*in*(erf(in*(FLOAT4)(0.707106793288165f))+(FLOAT4)(1.f))";
    build_options.emplace(" -DOPERATOR=" + compute);
    return build_options;
}


OpenCLGeluLayerAcc::~OpenCLGeluLayerAcc() {}

REGISTER_OPENCL_ACC(Gelu, LAYER_GELU)
REGISTER_OPENCL_LAYOUT(LAYER_GELU, DATA_FORMAT_NHC4W4)

}