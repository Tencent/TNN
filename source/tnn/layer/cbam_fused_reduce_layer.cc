#include <algorithm>
#include <cmath>

#include "tnn/layer/base_layer.h"

namespace TNN_NS {

DECLARE_LAYER(CbamFusedReduce, LAYER_CBAM_FUSED_REDUCE);

Status CbamFusedReduceLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status CbamFusedReduceLayer::InferOutputShape(bool ignore_error) {
    Blob* input_blob = input_blobs_[0];

    auto dims_input = input_blob->GetBlobDesc().dims;
    int num         = dims_input[0];
    int channels    = 2;
    int height      = dims_input[2];
    int width       = dims_input[3];

    DimsVector output_dims;
    output_dims.push_back(num);
    output_dims.push_back(channels);
    output_dims.push_back(height);
    output_dims.push_back(width);

    for (int i = 0; i < output_blobs_.size(); ++i) {
        output_blobs_[i]->GetBlobDesc().dims = output_dims;
    }
    return TNN_OK;
}

REGISTER_LAYER(CbamFusedReduce, LAYER_CBAM_FUSED_REDUCE);

}  // namespace TNN_NS
