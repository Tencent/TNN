#include "base.inc"

__kernel void GroupNorm(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t scale,
                        __read_only image2d_t bias, __private const int group, __private const float epsilon,
                        shape_4d output_shape, __write_only image2d_t output) {
    const int cw_idx = get_global_id(0);
    const int hb_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw_idx, hb_idx);

    const int channel           = output_shape.data[1];
    const int height            = output_shape.data[2];
    const int width             = output_shape.data[3];
    const int channel_per_group = channel / group;
    const int group_area        = height * width * channel_per_group;

    const int b_idx = hb_idx / height;
    const int h_idx = hb_idx % height;
    const int c_idx = cw_idx / width;
    const int w_idx = cw_idx % width;

    float sum_x             = 0;
    float sum_x2            = 0;
    const int channel_index = c_idx << 2;
    const int channel_start = (channel_index / channel_per_group * channel_per_group) >> 2;
    const int channel_end   = channel_start + (channel_per_group >> 2);
    float4 in;
    for (int w = 0; w < width; w++) {
        for (int c = channel_start; c < channel_end; c++) {
            for (int h = 0; h < height; h++) {
                in = read_imagef(input, SAMPLER, (int2)(c * width + w, b_idx * height + h));
                sum_x += in.x + in.y + in.z + in.w;
                sum_x2 += in.x * in.x + in.y * in.y + in.z * in.z + in.w * in.w;
            }
        }
    }

    float mean_x   = sum_x / (float)(group_area);
    float mean_x2  = sum_x2 / (float)(group_area);
    float variance = mean_x2 - mean_x * mean_x;
    variance       = 1.0f / sqrt(variance + epsilon);

    float4 scale_data = 0;
    float4 tmp;
    {
        tmp          = read_imagef(scale, SAMPLER, (int2)(0, channel_index));
        scale_data.x = tmp.x;
        tmp          = read_imagef(scale, SAMPLER, (int2)(0, channel_index + 1));
        scale_data.y = tmp.x;
        tmp          = read_imagef(scale, SAMPLER, (int2)(0, channel_index + 2));
        scale_data.z = tmp.x;
        tmp          = read_imagef(scale, SAMPLER, (int2)(0, channel_index + 3));
        scale_data.w = tmp.x;
    }

    float4 bias_data = 0;
    {
        tmp         = read_imagef(bias, SAMPLER, (int2)(0, channel_index));
        bias_data.x = tmp.x;
        tmp         = read_imagef(bias, SAMPLER, (int2)(0, channel_index + 1));
        bias_data.y = tmp.x;
        tmp         = read_imagef(bias, SAMPLER, (int2)(0, channel_index + 2));
        bias_data.z = tmp.x;
        tmp         = read_imagef(bias, SAMPLER, (int2)(0, channel_index + 3));
        bias_data.w = tmp.x;
    }

    in         = read_imagef(input, SAMPLER, (int2)(cw_idx, hb_idx));
    float4 out = (in - mean_x) * variance * scale_data + bias_data;

    write_imagef(output, (int2)(cw_idx, hb_idx), out);
}
