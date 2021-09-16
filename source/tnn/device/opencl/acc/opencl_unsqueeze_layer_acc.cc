#include "tnn/device/opencl/acc/opencl_layer_acc.h"

namespace TNN_NS {
class OpenCLUnsqueezeLayerAcc: public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;
    
    virtual ~OpenCLUnsqueezeLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type) override;

    std::shared_ptr<cl::Buffer> inter_buffer_ = nullptr;
};

Status OpenCLUnsqueezeLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Unsqueeze Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret);

    // ?
    run_3d_ndrange_ = false;
    op_name_        = "Unsqueeze";

    execute_units_.resize(2);
    // image->buffer
    {
        ret = CreateExecuteUnit(execute_units_[0], "image_to_buffer", "ImageToNCHWBuffer");
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    // buffer->image
    {
        ret = CreateExecuteUnit(execute_units_[1], "buffer_to_image", "NCHWBufferToImage");
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLUnsqueezeLayerAcc::~OpenCLUnsqueezeLayerAcc() {}

Status OpenCLUnsqueezeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Unsqueeze Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret);

    auto input = inputs[0];
    auto output = outputs[0];

    auto input_dims = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    int size0          = UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4) * 4 * DimsFunctionUtils::GetDim(output_dims, 0) *
                                DimsFunctionUtils::GetDim(output_dims, 2) * DimsFunctionUtils::GetDim(output_dims, 3);
    int size1          = UP_DIV(DimsFunctionUtils::GetDim(input_dims, 1), 4) * 4 * DimsFunctionUtils::GetDim(input_dims, 0) *
                                DimsFunctionUtils::GetDim(input_dims, 2) * DimsFunctionUtils::GetDim(input_dims, 3);
    int blob_size      = std::max(size0, size1) * sizeof(float);

    inter_buffer_      = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, blob_size);

    // image->buffer
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], input_dims);
        execute_units_[0].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
    }

    // buffer->image
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[1], output_dims);
        execute_units_[1].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
        execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
        execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    }

    return TNN_OK;
}

std::vector<DataType> OpenCLUnsqueezeLayerAcc::SupportDataType(int dims_size, BlobType blob_type) {
    if (blob_type == BLOB_INPUT) {
        return {DATA_TYPE_FLOAT, DATA_TYPE_HALF, DATA_TYPE_INT32};
    } else {
        return {DATA_TYPE_INT32};
    }
}

// REGISTER_OPENCL_ACC(Unsqueeze, LAYER_UNSQUEEZE)
// REGISTER_OPENCL_LAYOUT(LAYER_UNSQUEEZE, DATA_FORMAT_NHC4W4);

}