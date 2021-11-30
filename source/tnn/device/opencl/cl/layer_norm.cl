#include "base.inc"


__kernel void LayerNormDim3Reduce1D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t scale,
                                    __read_only image2d_t bias, __private const float eps, __private const int height,
                                    __write_only image2d_t output) {
    const int width_idx    = get_global_id(0);
    const int chan_blk_idx = get_global_id(1);
    const int hb_idx       = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, chan_blk_idx, hb_idx);
    const int width     = global_size_dim0;
    const int out_h_idx = hb_idx / height;

    int pos = mad24(chan_blk_idx, width, width_idx);

    FLOAT4 in;
    FLOAT4 mean  = (FLOAT4)0;
    FLOAT4 var   = (FLOAT4)0;
    int hb_start = out_h_idx * height;
    for (int h = 0; h < height; h++) {
        in = RI_F(input, SAMPLER, (int2)(pos, hb_start + h));
        mean += in;
        var += (in * in);
    }
    mean /= height;
    var = var / height - mean * mean;
    var = ((FLOAT4)(1.0f)) / sqrt(var + ((FLOAT4)(eps)));
    FLOAT4 data   = RI_F(input, SAMPLER, (int2)(pos, hb_idx));
    FLOAT4 scale_ = (FLOAT4)RI_F(scale, SAMPLER, (int2)(0, hb_idx % height)).s0;
    FLOAT4 bias_  = (FLOAT4)RI_F(bias, SAMPLER, (int2)(0, hb_idx % height)).s0;

    data -= mean;
    scale_ *= var;
    data = mad(data, scale_, bias_);

    WI_F(output, (int2)(pos, hb_idx), data);
}
