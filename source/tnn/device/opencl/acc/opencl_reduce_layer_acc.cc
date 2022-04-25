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

#include "tnn/device/opencl/acc/opencl_reduce_layer_acc.h"

#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

#define LowOpParallelismThre 256
#define HighOpIntensityThre 128

Status OpenCLReduceLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Reduce Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    auto reduce_param = dynamic_cast<ReduceLayerParam *>(param);
    if (!reduce_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    if (input_dims.size() > 4 && reduce_param->axis.size() > 1) {
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR,
                      "opencl reshape layer inputs not support dims > 4 and axis dims > 1");
    }

    int hb = DimsFunctionUtils::GetDim(output_dims, 0) * DimsFunctionUtils::GetDim(output_dims, 2);
    int cw = DimsFunctionUtils::GetDim(output_dims, 3) * UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4);

    if (reduce_param->axis.size() == 1) {
        int axis     = reduce_param->axis[0];
        axis         = axis >= 0 ? axis : axis + (int)input_dims.size();
        single_axis_ = axis;

        int axis_n = DimsFunctionUtils::GetDim(input_dims, axis);

        run_local_work_ = cw * hb < LowOpParallelismThre && axis_n >= HighOpIntensityThre;

        run_3d_ndrange_ = false;
        std::string kernel_name;
        if (axis == 0) {
            kernel_name = "ReduceC0";
        } else if (axis == 1) {
            kernel_name = "ReduceC1";
        } else if (axis == 2) {
            kernel_name = "ReduceC2";
        } else {
            kernel_name = "ReduceC3";
        }

        if (run_local_work_) {
            kernel_name += "Local";
        }

        if (input_dims.size() > 4) {
            kernel_name = "ReduceC2";
        }

        std::set<std::string> build_options = CreateBuildOptions();
        build_options.insert(build_options_.begin(), build_options_.end());

        ret = CreateExecuteUnit(execute_units_[0], "reduce", kernel_name, build_options);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    } else {
        run_3d_ndrange_         = false;
        std::string kernel_name = "ReduceMultiAxis";

        std::set<std::string> build_options = CreateBuildOptions();
        build_options.insert(build_options_.begin(), build_options_.end());

        ret = CreateExecuteUnit(execute_units_[0], "reduce", kernel_name, build_options);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLReduceLayerAcc::~OpenCLReduceLayerAcc() {}

Status OpenCLReduceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Reduce Layer Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto reduce_param = dynamic_cast<ReduceLayerParam *>(param_);
    if (!reduce_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    ASSERT(inputs.size() == 1);

    input_need_reshape_  = inputs[0]->GetBlobDesc().dims.size() > 4;
    output_need_reshape_ = (reduce_param->keep_dims == 0) || input_need_reshape_;

    // init input_reshape
    if (input_need_reshape_) {
        single_axis_ = reduce_param->axis[0] == 0 ? 0 : 2;
        auto shape   = GenerateInputShape(inputs[0]->GetBlobDesc().dims, reduce_param->axis[0]);
        ret          = CreaterBlob(inputs[0]->GetBlobDesc(), shape, reduce_input_blob_);
        CHECK_TNN_OK(ret)

        reduce_inputs_.clear();
        reduce_inputs_.push_back(reduce_input_blob_.get());
        ret = InitReshapeLayer(inputs, reduce_inputs_, shape, reshape_input_layer_acc_);
        CHECK_TNN_OK(ret)
    }

    auto reduce_input_blob = (input_need_reshape_ ? reduce_inputs_ : inputs)[0];
    auto reduce_input_dims = reduce_input_blob->GetBlobDesc().dims;

    // init output_reshape
    if (output_need_reshape_) {
        auto shape = outputs[0]->GetBlobDesc().dims;
        auto axis  = reduce_param->axis;
        if (input_need_reshape_) {
            axis = {single_axis_};
        }
        auto reduce_output_dims = AfterReduceDims(reduce_input_dims, axis);
        ret                     = CreaterBlob(outputs[0]->GetBlobDesc(), reduce_output_dims, reduce_output_blob_);
        CHECK_TNN_OK(ret)

        reduce_outputs_.clear();
        reduce_outputs_.push_back(reduce_output_blob_.get());
        ret = InitReshapeLayer(reduce_outputs_, outputs, shape, reshape_output_layer_acc_);
        CHECK_TNN_OK(ret)
    }

    auto reduce_output_blob = (output_need_reshape_ ? reduce_outputs_ : outputs)[0];
    auto reduce_output_dims = reduce_output_blob->GetBlobDesc().dims;

    int hb = DimsFunctionUtils::GetDim(reduce_output_dims, 0) * DimsFunctionUtils::GetDim(reduce_output_dims, 2);
    int cw =
        DimsFunctionUtils::GetDim(reduce_output_dims, 3) * UP_DIV(DimsFunctionUtils::GetDim(reduce_output_dims, 1), 4);
    int c4_n = DimsFunctionUtils::GetDim(reduce_input_dims, 1) / 4;
    int c4_r = DimsFunctionUtils::GetDim(reduce_input_dims, 1) % 4;
    int cw4  = DimsFunctionUtils::GetDim(reduce_input_dims, 3) * c4_n;

    if (input_need_reshape_) {
        if (reshape_input_layer_acc_ == nullptr) {
            return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "reshape layer acc in Reduce is null");
        }
        ret = reshape_input_layer_acc_->Reshape(inputs, reduce_inputs_);
        CHECK_TNN_OK(ret)
    }

    if (reduce_param->axis.size() == 1) {
        int axis = single_axis_;
        axis     = axis >= 0 ? axis : axis + (int)reduce_input_dims.size();

        int axis_n = DimsFunctionUtils::GetDim(reduce_input_dims, axis);

        auto &unit              = execute_units_[0];
        uint32_t workgroup_size = 0;

        OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
        int type_size                 = sizeof(float);
        if (opencl_runtime->GetPrecision() != PRECISION_HIGH) {
            type_size = 2;
        }

        if (run_local_work_) {
            workgroup_size =
                std::min(static_cast<uint32_t>(unit.local_mem_size / (4 * type_size)), unit.workgroupsize_max);
            workgroup_size = std::min(static_cast<uint32_t>(axis == 1 ? c4_n : axis_n), workgroup_size);
            int temp_size  = 1;
            while ((temp_size <<= 1) <= workgroup_size)
                ;
            workgroup_size = temp_size >> 1;

            unit.global_work_size = {static_cast<uint32_t>(cw * workgroup_size), static_cast<uint32_t>(hb)};
            unit.local_work_size  = {workgroup_size, 1};
        } else {
            unit.global_work_size = {static_cast<uint32_t>(cw), static_cast<uint32_t>(hb)};
            unit.local_work_size  = LocalWS2DDefault(unit);
        }

        uint32_t idx = 0;
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);

        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reduce_input_blob->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reduce_output_blob->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(reduce_input_dims, 0));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(reduce_input_dims, 1));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(reduce_input_dims, 2));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(reduce_input_dims, 3));
        execute_units_[0].ocl_kernel.setArg(idx++, c4_n);
        execute_units_[0].ocl_kernel.setArg(idx++, c4_r);
        execute_units_[0].ocl_kernel.setArg(idx++, cw4);
        execute_units_[0].ocl_kernel.setArg(idx++, axis_n);

        if (run_local_work_) {
            if (axis == 1) {
                execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(c4_n, workgroup_size));
            } else {
                execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(axis_n, workgroup_size));
            }
            execute_units_[0].ocl_kernel.setArg(idx++, workgroup_size * 4 * type_size, nullptr);
        }
    } else {
        auto &unit              = execute_units_[0];
        uint32_t workgroup_size = 0;

        int axis_n                 = 1;
        std::vector<int> axis_nhwc = {0, 0, 0, 0};
        for (int i = 0; i < reduce_param->axis.size(); i++) {
            int axis = reduce_param->axis[i];
            axis     = axis >= 0 ? axis : axis + (int)reduce_input_dims.size();
            switch (axis) {
                case 0:
                    if (!axis_nhwc[0]) {
                        axis_n *= DimsFunctionUtils::GetDim(reduce_input_dims, axis);
                        axis_nhwc[0] = 1;
                    }
                    break;
                case 1:
                    if (!axis_nhwc[3]) {
                        axis_n *= DimsFunctionUtils::GetDim(reduce_input_dims, axis);
                        axis_nhwc[3] = 1;
                    }
                    break;
                case 2:
                    if (!axis_nhwc[1]) {
                        axis_n *= DimsFunctionUtils::GetDim(reduce_input_dims, axis);
                        axis_nhwc[1] = 1;
                    }
                    break;
                case 3:
                    if (!axis_nhwc[2]) {
                        axis_n *= DimsFunctionUtils::GetDim(reduce_input_dims, axis);
                        axis_nhwc[2] = 1;
                    }
                    break;
            }
        }

        unit.global_work_size = {static_cast<uint32_t>(cw), static_cast<uint32_t>(hb)};
        unit.local_work_size  = LocalWS2DDefault(unit);

        uint32_t idx = 0;
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);

        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reduce_input_blob->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reduce_output_blob->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(reduce_input_dims, 0));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(reduce_input_dims, 1));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(reduce_input_dims, 2));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(reduce_input_dims, 3));
        execute_units_[0].ocl_kernel.setArg(idx++, c4_n);
        execute_units_[0].ocl_kernel.setArg(idx++, c4_r);
        execute_units_[0].ocl_kernel.setArg(idx++, cw4);
        execute_units_[0].ocl_kernel.setArg(idx++, axis_n);
        execute_units_[0].ocl_kernel.setArg(idx++, 4 * sizeof(int), axis_nhwc.data());
    }

    // reshape
    if (output_need_reshape_) {
        if (reshape_output_layer_acc_ == nullptr) {
            return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "reshape layer acc in Reduce is null");
        }
        ret = reshape_output_layer_acc_->Reshape(reduce_outputs_, outputs);
        CHECK_TNN_OK(ret)
    }

    return TNN_OK;
}

Status OpenCLReduceLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = TNN_OK;

    if (input_need_reshape_) {
        if (reshape_input_layer_acc_ == nullptr) {
            return Status(TNNERR_OPENCL_ACC_FORWARD_ERROR, "reshape layer acc in Reduce is null");
        }
        ret = reshape_input_layer_acc_->Forward(inputs, reduce_inputs_);
        CHECK_TNN_OK(ret)
    }

    ret = OpenCLLayerAcc::Forward(inputs, outputs);

    if (output_need_reshape_) {
        if (reshape_output_layer_acc_ == nullptr) {
            return Status(TNNERR_OPENCL_ACC_FORWARD_ERROR, "reshape layer acc in Reduce is null");
        }
        ret = reshape_output_layer_acc_->Forward(reduce_outputs_, outputs);
        CHECK_TNN_OK(ret)
    }

    return ret;
}

Status OpenCLReduceLayerAcc::InitReshapeLayer(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                              DimsVector &reshape_shape,
                                              shared_ptr<OpenCLReshapeLayerAcc> &reshape_layer_acc) {
    Status ret = TNN_OK;

    reshape_layer_acc = std::make_shared<OpenCLReshapeLayerAcc>();
    if (reshape_layer_acc == nullptr) {
        LOGE("Create Reshape Layer Acc in InnerProduct failed!\n");
        return Status(TNNERR_CREATE_LAYER, "Create Reshape Layer Acc in Reduce failed!");
    }

    auto reduce_param = dynamic_cast<ReduceLayerParam *>(param_);
    if (!reduce_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    // Init LayerAcc
    std::string suffix                               = input_need_reshape_ ? "_Input" : "_Output";
    std::shared_ptr<ReshapeLayerParam> reshape_param = std::make_shared<ReshapeLayerParam>();
    reshape_param->name                              = layer_name_ + "_Reshape" + suffix;
    reshape_param->reshape_type                      = 0;
    reshape_param->axis                              = 0;
    reshape_param->num_axes                          = reshape_shape.size();
    reshape_param->shape                             = reshape_shape;
    reshape_layer_acc->Init(ocl_context_, reshape_param.get(), nullptr, inputs, outputs);

    reshape_param_vec_.emplace_back(reshape_param);

    return ret;
}

Status OpenCLReduceLayerAcc::CreaterBlob(BlobDesc desc, DimsVector dims, std::shared_ptr<Blob> &blob) {
    BlobDesc blob_desc    = desc;
    blob_desc.data_format = DATA_FORMAT_NHC4W4;
    blob_desc.dims        = dims;
    blob                  = std::make_shared<Blob>(blob_desc, true);
    if (blob == nullptr) {
        LOGE("Create reshape output blob in MatMul failed!\n");
        return Status(TNNERR_CREATE_LAYER, "Create reshape blob in Reduce failed!");
    }

    return TNN_OK;
}

DimsVector OpenCLReduceLayerAcc::GenerateInputShape(DimsVector &input_dims, int axis) {
    const int batch           = DimsFunctionUtils::GetDim(input_dims, 0);
    const int channel_count   = DimsVectorUtils::Count(input_dims, 1, axis);
    const int axis_dims       = DimsFunctionUtils::GetDim(input_dims, axis);
    const int last_dims_count = DimsVectorUtils::Count(input_dims, axis + 1);

    if (axis == 0) {
        return {batch, DimsVectorUtils::Count(input_dims, 1), 1, 1};
    }

    return {batch, channel_count, axis_dims, last_dims_count};
}

DimsVector OpenCLReduceLayerAcc::AfterReduceDims(DimsVector dims, std::vector<int> axis) {
    for (const auto item : axis) {
        dims[item] = 1;
    }

    return dims;
}

}  // namespace TNN_NS
