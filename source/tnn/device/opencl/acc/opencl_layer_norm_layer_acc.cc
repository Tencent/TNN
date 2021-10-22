#include <vector>

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

class OpenCLLayerNormLayerAcc: public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;
    
    virtual ~OpenCLLayerNormLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    Status ConvertWeights(float *weights_data_ptr, int output_channel);

private:
    bool share_channel_ = false;
    int reduce_dim_size_ = -1;
    float eps_;
};

Status OpenCLLayerNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init LayerNorm Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret);

    run_3d_ndrange_ = true;
    op_name_        = "LayerNorm";

    reduce_dim_size_ = dynamic_cast<LayerNormLayerParam *>(param)->reduce_dims_size;
    eps_             = dynamic_cast<LayerNormLayerParam *>(param)->eps;


    auto dims = inputs[0]->GetBlobDesc().dims;
   
    // only support dim size smaller than 4
    if(dims.size() != 3 || reduce_dim_size_ != 1) {
        return Status(TNNERR_PARAM_ERR, "Error: only support data dim size 3, and reduce dim size 1 \n");
    }
    std::string kernel_name = "LayerNormDim3Reduce1D";
    ret                     = CreateExecuteUnit(execute_units_[0], "layer_norm", kernel_name);
    if(ret != TNN_OK) {
        LOGD("create execute unit failed!\n");
        return ret;
    }
    
    return TNN_OK;
}

OpenCLLayerNormLayerAcc::~OpenCLLayerNormLayerAcc() {}

Status OpenCLLayerNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("LayerNorm Layer Reshape\n");
    ASSERT(inputs.size() == 3)
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret);

    auto input_dims = inputs[0]->GetBlobDesc().dims;
    int channels    = DimsFunctionUtils::GetDim(input_dims, 1);
    
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    uint32_t idx = SetExecuteUnit3DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[1]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[2]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, eps_);
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 2));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    return TNN_OK;
}

// ?
REGISTER_OPENCL_ACC(LayerNorm, LAYER_LAYER_NORM)
REGISTER_OPENCL_LAYOUT(LAYER_LAYER_NORM, DATA_FORMAT_NHC4W4);

}