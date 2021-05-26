#include "base.inc"

__kernel void Blob5DConvertFromNCHW(GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
                                    __global const float *input_ptr, __private const int dim2,
                                    __private const int dim3, __private const int dim4,
                                    __private const int channels,
                                    __global const float* scale,
                                    __global const float* bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int dim3_idx       = image_height_idx % dim3;
    const int batch_dim2_idx = image_height_idx / dim3;
    const int dim2_idx       = batch_dim2_idx % dim2;
    const int batch_idx      = batch_dim2_idx / dim2;
    const int dim4_idx       = image_width_idx % dim4;
    const int channel_4_idx  = (image_width_idx / dim4) << 2;
    const int buffer_offset  = (((batch_idx * channels + channel_4_idx) * dim2 + dim2_idx) *
                                dim3 + dim3_idx) * dim4 + dim4_idx;

    const int stride         = dim2 * dim3 * dim4;
    const int remain_channel = channels - channel_4_idx;

    float4 output_values        = 0;

    if (remain_channel >= 4) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += stride;
        output_values.y = *(input_ptr + offset);
        offset += stride;
        output_values.z = *(input_ptr + offset);
        offset += stride;
        output_values.w = *(input_ptr + offset);
#ifdef ENABLE_SCALE_BIAS
        float4 scale_data   = vload4(0, scale + channel_4_idx);
        float4 bias_data    = vload4(0, bias + channel_4_idx);
        output_values = output_values * scale_data + bias_data;
#endif
    } else if (remain_channel == 3) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += stride;
        output_values.y = *(input_ptr + offset);
        offset += stride;
        output_values.z = *(input_ptr + offset);
#ifdef ENABLE_SCALE_BIAS
        float3 scale_data   = vload3(0, scale + channel_4_idx);
        float3 bias_data    = vload3(0, bias + channel_4_idx);
        output_values.xyz = output_values.xyz * scale_data + bias_data;
#endif
    } else if (remain_channel == 2) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += stride;
        output_values.y = *(input_ptr + offset);
#ifdef ENABLE_SCALE_BIAS
        float2 scale_data   = vload2(0, scale + channel_4_idx);
        float2 bias_data    = vload2(0, bias + channel_4_idx);
        output_values.xy = output_values.xy * scale_data + bias_data;
#endif
    } else if (remain_channel == 1) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
#ifdef ENABLE_SCALE_BIAS
        output_values.x = output_values.x * scale[channel_4_idx] + bias[channel_4_idx];
#endif
    }

    write_imagef(output, (int2)(image_width_idx, image_height_idx),
                 output_values);
}
