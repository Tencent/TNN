#include "base.inc"

__kernel void PadConst(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int output_height,
                       __private const int input_channel,
                       __private const int input_height,
                       __private const int input_width,
                       __private const int pad_l, __private const int pad_t,
                       __private const int pad_c_b, __private float pad_value) {
    const int output_w_idx      = get_global_id(0);
    const int channel_block_idx = get_global_id(1);
    const int output_hb_idx     = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_w_idx, channel_block_idx, output_hb_idx);
    const int b_idx        = output_hb_idx / output_height;
    const int output_h_idx = output_hb_idx % output_height;
    const int output_width = global_size_dim0;
    const int2 output_pos  = (int2)(
        mad24(channel_block_idx, output_width, output_w_idx), output_hb_idx);

    FLOAT4 val      = (FLOAT4)pad_value;
    int input_w_idx = output_w_idx - pad_l;
    int input_h_idx = output_h_idx - pad_t;
    int remain      = pad_c_b % 4;

    int input_c_b_idx = channel_block_idx * 4 - pad_c_b;
    int input_c_e_idx = input_c_b_idx + 3;
    int input_channel_b_block_idx = channel_block_idx - (pad_c_b + 3) / 4;
    int input_channel_e_block_idx = input_channel_b_block_idx + 1;
    if (input_w_idx >= 0 && input_w_idx < input_width && input_h_idx >= 0 &&
        input_h_idx < input_height && input_c_b_idx < input_channel && input_c_e_idx >= 0) {
        if (remain == 0)
        {
            const int2 input_pos =
                (int2)(mad24(input_channel_b_block_idx, input_width, input_w_idx),
                       mad24(b_idx, input_height, input_h_idx));
            val = RI_F(input, SAMPLER, input_pos);

            int remain_channels = input_channel - input_c_b_idx;

            if (remain_channels <= 3)
            {
                if (remain_channels == 3)
                {
                    val.w = pad_value;
                }
                else if (remain_channels == 2)
                {
                    val.z = val.w = pad_value;
                }
                else
                {
                    val.y = val.z = val.w = pad_value;
                }
            }
        }
        else
        {
            if (input_c_b_idx >= 0 && input_c_e_idx < input_channel)
            {
                const int2 input_b_block_pos =
                    (int2)(mad24(input_channel_b_block_idx, input_width, input_w_idx),
                           mad24(b_idx, input_height, input_h_idx));
                FLOAT4 temp_b_value = RI_F(input, SAMPLER, input_b_block_pos);

                const int2 input_e_block_pos =
                    (int2)(mad24(input_channel_e_block_idx, input_width, input_w_idx),
                           mad24(b_idx, input_height, input_h_idx));
                FLOAT4 temp_e_value = RI_F(input, SAMPLER, input_e_block_pos);

                if (remain == 1)
                {
                    val = (float4)(temp_b_value.w, temp_e_value.x, temp_e_value.y, temp_e_value.z);
                }
                else if (remain == 2)
                {
                    val = (float4)(temp_b_value.z, temp_b_value.w, temp_e_value.x, temp_e_value.y);
                }
                else
                {
                    val = (float4)(temp_b_value.y, temp_b_value.z, temp_b_value.w, temp_e_value.x);
                }
            }
            else if (input_c_b_idx < 0)
            {
                const int2 input_block_pad_pos =
                    (int2)(mad24(0, input_width, input_w_idx),
                           mad24(b_idx, input_height, input_h_idx));
                FLOAT4 temp_value = RI_F(input, SAMPLER, input_block_pad_pos);

                if (remain == 1)
                {
                    if (input_channel >= 3)
                        val = (float4)(pad_value, temp_value.x, temp_value.y, temp_value.z);
                    else if (input_channel >= 2)
                        val = (float4)(pad_value, temp_value.x, temp_value.y, pad_value);
                    else
                        val = (float4)(pad_value, temp_value.x, pad_value, pad_value);
                }
                else if (remain == 2)
                {
                    if (input_channel >= 2)
                        val = (float4)(pad_value, pad_value, temp_value.x, temp_value.y);
                    else
                        val = (float4)(pad_value, pad_value, temp_value.x, pad_value);
                }
                else
                {
                    val = (float4)(pad_value, pad_value, pad_value, temp_value.x);
                }
            }
            else
            {
                const int2 input_b_block_pos =
                    (int2)(mad24(input_channel_b_block_idx, input_width, input_w_idx),
                           mad24(b_idx, input_height, input_h_idx));
                FLOAT4 temp_b_value = RI_F(input, SAMPLER, input_b_block_pos);

                const int2 input_e_block_pos =
                    (int2)(mad24(input_channel_e_block_idx, input_width, input_w_idx),
                           mad24(b_idx, input_height, input_h_idx));
                FLOAT4 temp_e_value = RI_F(input, SAMPLER, input_e_block_pos);

                int remain_channels = input_channel - input_c_b_idx;

                if (remain == 1)
                {
                    if (remain_channels >= 4)
                        val = (float4)(temp_b_value.w, temp_e_value.x, temp_e_value.y, temp_e_value.z);
                    else if (remain_channels == 3)
                        val = (float4)(temp_b_value.w, temp_e_value.x, temp_e_value.y, pad_value);
                    else if (remain_channels == 2)
                        val = (float4)(temp_b_value.w, temp_e_value.x, pad_value, pad_value);
                    else
                        val = (float4)(temp_b_value.w, pad_value, pad_value, pad_value);
                }
                else if (remain == 2)
                {
                    if (remain_channels >= 4)
                        val = (float4)(temp_b_value.z, temp_b_value.w, temp_e_value.x, temp_e_value.y);
                    else if (remain_channels == 3)
                        val = (float4)(temp_b_value.z, temp_b_value.w, temp_e_value.x, pad_value);
                    else if (remain_channels == 2)
                        val = (float4)(temp_b_value.z, temp_b_value.w, pad_value, pad_value);
                    else
                        val = (float4)(temp_b_value.z, pad_value, pad_value, pad_value);
                }
                else
                {
                    if (remain_channels >= 4)
                        val = (float4)(temp_b_value.y, temp_b_value.z, temp_b_value.w, temp_e_value.x);
                    else if (remain_channels == 3)
                        val = (float4)(temp_b_value.y, temp_b_value.z, temp_b_value.w, pad_value);
                    else if (remain_channels == 2)
                        val = (float4)(temp_b_value.y, temp_b_value.z, pad_value, pad_value);
                    else
                        val = (float4)(temp_b_value.y, pad_value, pad_value, pad_value);
                }
            }
        }
    }

    WI_F(output, output_pos, val);
}

__kernel void PadReflect(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                         __write_only image2d_t output,
                         __private const int output_height,
                         __private const int input_channel,
                         __private const int input_height,
                         __private const int input_width,
                         __private const int pad_l, __private const int pad_t,
                         __private const int pad_c_b, __private float pad_value) {
    // NOTE: reflect mode not support channel level padding for now
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
