#include "base.inc"

__kernel void SoftmaxHeight(__read_only image2d_t input, __write_only image2d_t output,
                      __private const int4 shape // NCHW
                      ) {
    int wc = get_global_id(0);
    int b = get_global_id(1);
    if (wc < shape.y*shape.w && b < shape.x) {
        /*Compute Max */
        FLOAT4 maxValue = RI_F(input, SAMPLER, (int2)(wc, b*shape.z));
        for (int i=1; i<shape.z; ++i) {
            maxValue = fmax(maxValue, RI_F(input, SAMPLER, (int2)(wc, b*shape.z+i)));
        }
        /*Compute Exp Sum*/
        FLOAT4 sumValue = (FLOAT4)0;
        for (int i=0; i<shape.z; ++i) {
            sumValue += exp(RI_F(input, SAMPLER, (int2)(wc, b*shape.z+i)) - maxValue);
        }
        /*Compute Result */
        for (int i=0; i<shape.z; ++i) {
            FLOAT4 value = exp(RI_F(input, SAMPLER, (int2)(wc, b*shape.z+i)) - maxValue) / sumValue;
            WI_F(output, (int2)(wc, b*shape.z+i), value);
        }
    }
}

__kernel void SoftmaxChannel(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __write_only image2d_t output, __private const int output_channels,
                              __private const int remain_channels) {

    const int channel_block_idx = get_global_id(0);
    const int width_idx    = get_global_id(1);
    const int batch_height_idx       = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(channel_block_idx, width_idx, batch_height_idx);

    // const int width     = global_size_dim1;

    FLOAT float_max_value = -FLT_MAX;
    FLOAT4 input_data;
    for (short i = 0; i < global_size_dim0 - 1; ++i) {
        input_data      = RI_F(input, SAMPLER, (int2)(width_idx + i * global_size_dim1, batch_height_idx));
        float_max_value = max(float_max_value, input_data.x);
        float_max_value = max(float_max_value, input_data.y);
        float_max_value = max(float_max_value, input_data.z);
        float_max_value = max(float_max_value, input_data.w);
    }

    input_data = RI_F(input, SAMPLER, (int2)(width_idx + (global_size_dim0 - 1) * global_size_dim1 , batch_height_idx));
    if (remain_channels == 0) {
        float_max_value = max(float_max_value, input_data.w);
        float_max_value = max(float_max_value, input_data.z);
        float_max_value = max(float_max_value, input_data.y);
        float_max_value = max(float_max_value, input_data.x);
    } else if (remain_channels == 1) {
        float_max_value = max(float_max_value, input_data.z);
        float_max_value = max(float_max_value, input_data.y);
        float_max_value = max(float_max_value, input_data.x);
    } else if (remain_channels == 2) {
        float_max_value = max(float_max_value, input_data.y);
        float_max_value = max(float_max_value, input_data.x);
    } else if (remain_channels == 3) {
        float_max_value = max(float_max_value, input_data.x);
    }

    FLOAT accum_result       = 0;
    for (short i = 0; i < global_size_dim0 - 1; ++i) {
        input_data = RI_F(input, SAMPLER, (int2)(width_idx + i * global_size_dim1, batch_height_idx));
        input_data = exp(input_data - float_max_value);
        accum_result += input_data.x;
        accum_result += input_data.y;
        accum_result += input_data.z;
        accum_result += input_data.w;
    }

    input_data = RI_F(input, SAMPLER, (int2)(width_idx + (global_size_dim0 - 1) * global_size_dim1, batch_height_idx));
    input_data -= float_max_value;
    if (remain_channels == 0) {
        accum_result += exp(input_data.w);
        accum_result += exp(input_data.z);
        accum_result += exp(input_data.y);
        accum_result += exp(input_data.x);
    } else if (remain_channels == 1) {
        accum_result += exp(input_data.z);
        accum_result += exp(input_data.y);
        accum_result += exp(input_data.x);
    } else if (remain_channels == 2) {
        accum_result += exp(input_data.y);
        accum_result += exp(input_data.x);
    } else if (remain_channels == 3) {
        accum_result += exp(input_data.x);
    }

    int cur_out_width_pos  = mad24(channel_block_idx, global_size_dim1, width_idx);
    input_data = RI_F(input, SAMPLER, (int2)(cur_out_width_pos, batch_height_idx)) - float_max_value;
    const int output_remain = mul24(channel_block_idx, 4) - output_channels;

    if (output_remain == 1) {
        input_data.z = exp(input_data.z) / accum_result;
        input_data.y = exp(input_data.y) / accum_result;
        input_data.x = exp(input_data.x) / accum_result;
    } else if (output_remain == 2) {
        input_data.y = exp(input_data.y) / accum_result;
        input_data.x = exp(input_data.x) / accum_result;
    } else if (output_remain == 3) {
        input_data.x = exp(input_data.x) / accum_result;
    } else{
        input_data = exp(input_data) / accum_result;
    }

    WI_F(output, (int2)(cur_out_width_pos, batch_height_idx), input_data);
}
