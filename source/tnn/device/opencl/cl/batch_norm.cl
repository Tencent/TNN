#include "base.inc"

__kernel void BatchNorm(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                        __read_only image2d_t scale, __read_only image2d_t eps,
                        __private const int width,
                        __write_only image2d_t output) {
    const int cw_idx = get_global_id(0);
    const int hb_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw_idx, hb_idx);

    const int chan_blk_idx = cw_idx / width;

    float4 data   = read_imagef(input, SAMPLER, (int2)(cw_idx, hb_idx));
    float4 scale_ = read_imagef(scale, SAMPLER, (int2)(chan_blk_idx, 0));
    float4 eps_   = read_imagef(eps, SAMPLER, (int2)(chan_blk_idx, 0));
    data          = mad(data, scale_, eps_);

    write_imagef(output, (int2)(cw_idx, hb_idx), data);
}

__kernel void BatchNormBatch(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                             __read_only image2d_t scale,
                             __read_only image2d_t eps,
                             __private const int width,
                             __private const int height,
                             __write_only image2d_t output) {
    const int cw_idx = get_global_id(0);
    const int hb_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw_idx, hb_idx);

    const int chan_blk_idx = cw_idx / width;
    const int b_idx        = hb_idx / height;

    float4 data   = read_imagef(input, SAMPLER, (int2)(cw_idx, hb_idx));
    float4 scale_ = read_imagef(scale, SAMPLER, (int2)(chan_blk_idx, b_idx));
    float4 eps_   = read_imagef(eps, SAMPLER, (int2)(chan_blk_idx, b_idx));
    data          = mad(data, scale_, eps_);

    write_imagef(output, (int2)(cw_idx, hb_idx), data);
}

__kernel void BatchNormGS3D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                            __read_only image2d_t scale,
                            __read_only image2d_t eps,
                            __write_only image2d_t output) {
    const int width_idx    = get_global_id(0);
    const int chan_blk_idx = get_global_id(1);
    const int hb_idx       = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, chan_blk_idx, hb_idx);
    const int width = global_size_dim0;

    int pos = mad24(chan_blk_idx, width, width_idx);

    float4 data   = read_imagef(input, SAMPLER, (int2)(pos, hb_idx));
    float4 scale_ = read_imagef(scale, SAMPLER, (int2)(chan_blk_idx, 0));
    float4 eps_   = read_imagef(eps, SAMPLER, (int2)(chan_blk_idx, 0));
    data          = mad(data, scale_, eps_);

    write_imagef(output, (int2)(pos, hb_idx), data);
}
