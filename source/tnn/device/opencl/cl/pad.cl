#include "base.inc"

__kernel void PadConst(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int output_height,
                       __private const int input_height,
                       __private const int input_width,
                       __private const int pad_l, __private const int pad_t) {
    const int output_w_idx      = get_global_id(0);
    const int channel_block_idx = get_global_id(1);
    const int output_hb_idx     = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_w_idx, channel_block_idx, output_hb_idx);
    const int b_idx        = output_hb_idx / output_height;
    const int output_h_idx = output_hb_idx % output_height;
    const int output_width = global_size_dim0;
    const int2 output_pos  = (int2)(
        mad24(channel_block_idx, output_width, output_w_idx), output_hb_idx);

    FLOAT4 val      = (FLOAT4)0.0f;
    int input_w_idx = output_w_idx - pad_l;
    int input_h_idx = output_h_idx - pad_t;
    if (input_w_idx >= 0 && input_w_idx < input_width && input_h_idx >= 0 &&
        input_h_idx < input_height) {
        const int2 input_pos =
            (int2)(mad24(channel_block_idx, input_width, input_w_idx),
                   mad24(b_idx, input_height, input_h_idx));
        val = RI_F(input, SAMPLER, input_pos);
    }

    WI_F(output, output_pos, val);
}

__kernel void PadReflect(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                         __write_only image2d_t output,
                         __private const int output_height,
                         __private const int input_height,
                         __private const int input_width,
                         __private const int pad_l, __private const int pad_t) {
    const int output_w_idx      = get_global_id(0);
    const int channel_block_idx = get_global_id(1);
    const int output_hb_idx     = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_w_idx, channel_block_idx, output_hb_idx);
    const int b_idx        = output_hb_idx / output_height;
    const int output_h_idx = output_hb_idx % output_height;
    const int output_width = global_size_dim0;

    int input_w_idx = output_w_idx - pad_l;
    if (input_w_idx < 0) {
        input_w_idx = -input_w_idx;
    } else if (input_w_idx >= input_width) {
        input_w_idx = input_width - (input_w_idx - input_width) - 2;
    }
    int input_h_idx = output_h_idx - pad_t;
    if (input_h_idx < 0) {
        input_h_idx = -input_h_idx;
    } else if (input_h_idx >= input_height) {
        input_h_idx = input_height - (input_h_idx - input_height) - 2;
    }

    const int2 input_pos =
        (int2)(mad24(channel_block_idx, input_width, input_w_idx),
               mad24(b_idx, input_height, input_h_idx));
    const int2 output_pos = (int2)(
        mad24(channel_block_idx, output_width, output_w_idx), output_hb_idx);

    WI_F(output, output_pos, RI_F(input, SAMPLER, input_pos));
}
