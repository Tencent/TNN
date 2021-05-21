#include "base.inc"

__kernel void Blob6DConvertToNCHW(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_ptr,
                                  __global float *output, __private const int dim2,
                                  __private const int dim3, __private const int dim4,
                                  __private const int dim5, __private const int channels,
                                  __global const float* scale,
                                  __global const float* bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int dim3_idx       = image_height_idx % dim3;
    const int batch_dim2_idx = image_height_idx / dim3;
    const int dim2_idx       = batch_dim2_idx % dim2;
    const int batch_idx      = batch_dim2_idx / dim2;
    const int dim5_idx       = image_width_idx % dim5;
    const int channel_updiv_4_dim4_idx = image_width_idx / dim5;
    const int dim4_idx       = channel_updiv_4_dim4_idx % dim4;
    const int channel_4_idx  = (channel_updiv_4_dim4_idx / dim4) << 2;
    const int buffer_offset  = ((((batch_idx * channels + channel_4_idx) * dim2 + dim2_idx) *
                                dim3 + dim3_idx) * dim4 + dim4_idx) * dim5 + dim5_idx;

    const int stride         = dim2 * dim3 * dim4 * dim5;
    int2 coord               = (int2)(image_width_idx, image_height_idx);
    float4 values            = read_imagef(input_ptr, SAMPLER, coord);
    const int remain_channel = channels - channel_4_idx;

    if (remain_channel >= 4) {
#ifdef ENABLE_SCALE_BIAS
        float4 scale_data   = vload4(0, scale + channel_4_idx);
        float4 bias_data    = vload4(0, bias + channel_4_idx);
        values = values * scale_data + bias_data;
#endif
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += stride;
        output[offset] = values.y;
        offset += stride;
        output[offset] = values.z;
        offset += stride;
        output[offset] = values.w;
    } else if (remain_channel == 3) {
#ifdef ENABLE_SCALE_BIAS
        float3 scale_data   = vload3(0, scale + channel_4_idx);
        float3 bias_data    = vload3(0, bias + channel_4_idx);
        values.xyz = values.xyz * scale_data + bias_data;
#endif
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += stride;
        output[offset] = values.y;
        offset += stride;
        output[offset] = values.z;
    } else if (remain_channel == 2) {
#ifdef ENABLE_SCALE_BIAS
        float2 scale_data   = vload2(0, scale + channel_4_idx);
        float2 bias_data    = vload2(0, bias + channel_4_idx);
        values.xy = values.xy * scale_data + bias_data;
#endif
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += stride;
        output[offset] = values.y;
    } else if (remain_channel == 1) {
#ifdef ENABLE_SCALE_BIAS
        values.x = values.x * scale[channel_4_idx] + bias[channel_4_idx];
#endif
        int offset     = buffer_offset;
        output[offset] = values.x;
    }
}