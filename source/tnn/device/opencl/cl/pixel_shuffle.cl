#include "base.inc"

#define SET_CHANNEL_OUTPUT(in, in_c_idx, out)   \
    in_c_idx_remain = in_c_idx & 3;             \
    if (in_c_idx_remain == 0) {                 \
        out = in.x;                             \
    } else if (in_c_idx_remain == 1) {          \
        out = in.y;                             \
    } else if (in_c_idx_remain == 2) {          \
        out = in.z;                             \
    } else {                                    \
        out = in.w;                             \
    }

__kernel void PixelShuffle(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                           __write_only image2d_t output,
                           __private const int output_height,
                           __private const int output_width,
                           __private const int input_height,
                           __private const int input_width,
                           __private const int upscale_factor,
                           __private const int upscale_factor_pow) {
    const int out_cw_idx = get_global_id(0);
    const int out_bh_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(out_cw_idx, out_bh_idx);

    const int out_channel_blocks_idx    = out_cw_idx / output_width;
    const int out_width_idx             = out_cw_idx % output_width;
    const int out_batch_idx             = out_bh_idx / output_height;
    const int out_height_idx            = out_bh_idx % output_height;

    const int in_width_idx  = out_width_idx / upscale_factor;
    const int in_height_idx = out_height_idx / upscale_factor;

    const int out_channel_base  = out_channel_blocks_idx << 2;
    const int in_channel_offset = mad24(
            out_height_idx % upscale_factor, upscale_factor,
            out_width_idx % upscale_factor);
    
    const int in_channel_idx0 = mad24(out_channel_base, upscale_factor_pow, in_channel_offset);
    const int in_channel_idx1 = in_channel_idx0 + upscale_factor_pow;
    const int in_channel_idx2 = in_channel_idx1 + upscale_factor_pow;
    const int in_channel_idx3 = in_channel_idx2 + upscale_factor_pow;

    const int in_channel_blocks_idx0 = in_channel_idx0 >> 2;
    const int in_channel_blocks_idx1 = in_channel_idx1 >> 2;
    const int in_channel_blocks_idx2 = in_channel_idx2 >> 2;
    const int in_channel_blocks_idx3 = in_channel_idx3 >> 2;

    FLOAT4 in0, in1, in2, in3, out;
    int2 pos0, pos1, pos2, pos3;
    pos0.x = mad24(in_channel_blocks_idx0, input_width, in_width_idx);
    pos0.y = mad24(out_batch_idx, input_height, in_height_idx);
    pos1.x = mad24(in_channel_blocks_idx1, input_width, in_width_idx);
    pos1.y = pos0.y;
    pos2.x = mad24(in_channel_blocks_idx2, input_width, in_width_idx);
    pos2.y = pos0.y;
    pos3.x = mad24(in_channel_blocks_idx3, input_width, in_width_idx);
    pos3.y = pos0.y;

    in0 = RI_F(input, SAMPLER, pos0);
    in1 = RI_F(input, SAMPLER, pos1);
    in2 = RI_F(input, SAMPLER, pos2);
    in3 = RI_F(input, SAMPLER, pos3);

    int in_c_idx_remain;
    SET_CHANNEL_OUTPUT(in0, in_channel_idx0, out.x);
    SET_CHANNEL_OUTPUT(in1, in_channel_idx1, out.y);
    SET_CHANNEL_OUTPUT(in2, in_channel_idx2, out.z);
    SET_CHANNEL_OUTPUT(in3, in_channel_idx3, out.w);

    WI_F(output, (int2)(out_cw_idx, out_bh_idx), out);
}