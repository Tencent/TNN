#include "base.inc"

#define GET_RESULT_FROM_INPUT(V)                                               \
    input_val = RI_F(                                                          \
        input, SAMPLER,                                                        \
        (int2)(mad24(input_channel_block_idx.V, width, width_idx), hb_idx));   \
    if (input_channel_sub_idx.V == 0) {                                        \
        result.V = input_val.x;                                                \
    } else if (input_channel_sub_idx.V == 1) {                                 \
        result.V = input_val.y;                                                \
    } else if (input_channel_sub_idx.V == 2) {                                 \
        result.V = input_val.z;                                                \
    } else if (input_channel_sub_idx.V == 3) {                                 \
        result.V = input_val.w;                                                \
    }

__kernel void ShuffleChannel(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                             __write_only image2d_t output,
                             __private const int group,
                             __private const int group_size,
                             __private const int channels) {
    const int width_idx         = get_global_id(0);
    const int channel_block_idx = get_global_id(1);
    const int hb_idx            = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, channel_block_idx, hb_idx);
    const int width = global_size_dim0;

    int2 output_pos =
        (int2)(mad24(channel_block_idx, width, width_idx), hb_idx);
    int4 output_channel_idx = (channel_block_idx << 2) + (int4)(0, 1, 2, 3);
    int4 input_channel_idx =
        output_channel_idx % group * group_size + output_channel_idx / group;
    input_channel_idx            = min(input_channel_idx, (int4)(channels - 1));
    int4 input_channel_block_idx = input_channel_idx >> 2;
    int4 input_channel_sub_idx   = input_channel_idx % 4;

    FLOAT4 result = (FLOAT4)(0, 0, 0, 0);
    FLOAT4 input_val;

    GET_RESULT_FROM_INPUT(x);
    GET_RESULT_FROM_INPUT(y);
    GET_RESULT_FROM_INPUT(z);
    GET_RESULT_FROM_INPUT(w);

    WI_F(output, output_pos, result);
}
