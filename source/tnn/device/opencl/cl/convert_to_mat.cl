#include "base.inc"

__kernel void ConvertToNCHW(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_ptr,
                            __global float *output, __private const int height,
                            __private const int width,
                            __private const int channels,
                            __constant float* scale,
                            __constant float* bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    int buffer_offset =
        ((batch_idx * channels + channel_4_idx) * height + height_idx) * width +
        width_idx;
    float4 values = read_imagef(input_ptr, SAMPLER,
                                (int2)(image_width_idx, image_height_idx));

    const int height_width_size = height * width;

    const int remain_channel = channels - channel_4_idx;

    if (remain_channel >= 4) {
#ifdef ENABLE_SCALE_BIAS
        float4 scale_data   = vload4(0, scale + channel_4_idx);
        float4 bias_data    = vload4(0, bias + channel_4_idx);
        values = values * scale_data + bias_data;
#endif
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += height_width_size;
        output[offset] = values.y;
        offset += height_width_size;
        output[offset] = values.z;
        offset += height_width_size;
        output[offset] = values.w;
    } else if (remain_channel == 3) {
#ifdef ENABLE_SCALE_BIAS
        float3 scale_data   = vload3(0, scale + channel_4_idx);
        float3 bias_data    = vload3(0, bias + channel_4_idx);
        values.xyz = values.xyz * scale_data + bias_data;
#endif
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += height_width_size;
        output[offset] = values.y;
        offset += height_width_size;
        output[offset] = values.z;
    } else if (remain_channel == 2) {
#ifdef ENABLE_SCALE_BIAS
        float2 scale_data   = vload2(0, scale + channel_4_idx);
        float2 bias_data    = vload2(0, bias + channel_4_idx);
        values.xy = values.xy * scale_data + bias_data;
#endif
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += height_width_size;
        output[offset] = values.y;
    } else if (remain_channel == 1) {
#ifdef ENABLE_SCALE_BIAS
        values.x = values.x * scale[channel_4_idx] + bias[channel_4_idx];
#endif
        int offset     = buffer_offset;
        output[offset] = values.x;
    }
}

__kernel void ConvertToN8UC4(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_ptr,
                             __global uchar *output, __private const int height,
                             __private const int width,
                             __private const float4 scale,
                             __private const float4 bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx         = image_height_idx / height;
    const int height_idx        = image_height_idx % height;
    const int width_idx         = image_width_idx % width;
    const int channel_block_idx = image_width_idx / width;

    int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx +
                         channel_block_idx) *
                        4;

    int2 coord      = (int2)(image_width_idx, image_height_idx);
    float4 values_f = read_imagef(input_ptr, SAMPLER, coord);
#ifdef ENABLE_SCALE_BIAS
    values_f = values_f * scale + bias;
#endif
    uchar4 values = convert_uchar4_sat(values_f);

#ifdef SWAP_RB
    uchar temp = values.x;
    values.x   = values.z;
    values.z   = temp;
#endif

    vstore4(values, 0, output + buffer_offset);
}

__kernel void ConvertToN8UC3(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_ptr,
                             __global uchar *output, __private const int height,
                             __private const int width,
                             __private const float4 scale,
                             __private const float4 bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx         = image_height_idx / height;
    const int height_idx        = image_height_idx % height;
    const int width_idx         = image_width_idx % width;
    const int channel_block_idx = image_width_idx / width;

    int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx +
                         channel_block_idx) *
                        3;

    int2 coord      = (int2)(image_width_idx, image_height_idx);
    float4 values_f = read_imagef(input_ptr, SAMPLER, coord);
#ifdef ENABLE_SCALE_BIAS
    values_f = values_f * scale + bias;
#endif
    uchar4 values = convert_uchar4_sat(values_f);

#ifdef SWAP_RB
    uchar temp = values.x;
    values.x   = values.z;
    values.z   = temp;
#endif
    output[buffer_offset]     = values.x;
    output[buffer_offset + 1] = values.y;
    output[buffer_offset + 2] = values.z;
}

__kernel void ConvertToNGray(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_ptr,
                             __global uchar *output, __private const int height,
                             __private const int width,
                             __private const float4 scale,
                             __private const float4 bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx         = image_height_idx / height;
    const int height_idx        = image_height_idx % height;
    const int width_idx         = image_width_idx % width;
    const int channel_block_idx = image_width_idx / width;

    int buffer_offset = (batch_idx * height + height_idx) * width + width_idx +
                        channel_block_idx;

    int2 coord      = (int2)(image_width_idx, image_height_idx);
    float4 values_f = read_imagef(input_ptr, SAMPLER, coord);
#ifdef ENABLE_SCALE_BIAS
    values_f = values_f * scale + bias;
#endif
    uchar4 values = convert_uchar4_sat(values_f);

    output[buffer_offset] = values.x;
}

__kernel void ConvertToN32FC4Image(
    GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
    __read_only image2d_t input, __private const float4 scale,
    __private const float4 bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    int2 coord    = (int2)(image_width_idx, image_height_idx);
    float4 values = read_imagef(input, SAMPLER, coord);
#ifdef ENABLE_SCALE_BIAS
    values = values * scale + bias;
#endif

#ifdef SWAP_RB
    float temp = values.x;
    values.x   = values.z;
    values.z   = temp;
#endif

    write_imagef(output, coord, values);
}

__kernel void CopyToN8UC3(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_ptr,
                             __global uchar *output, __private const int height,
                             __private const int width) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx         = image_height_idx / height;
    const int height_idx        = image_height_idx % height;
    const int width_idx         = image_width_idx % width;
    const int channel_block_idx = image_width_idx / width;

    int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx +
                         channel_block_idx) *
                        3;

    int2 coord      = (int2)(image_width_idx, image_height_idx);
    float4 values_f = read_imagef(input_ptr, SAMPLER, coord);
    uchar4 values = convert_uchar4_sat(values_f);

    output[buffer_offset]     = values.x;
    output[buffer_offset + 1] = values.y;
    output[buffer_offset + 2] = values.z;
}

__kernel void CopyToN8UC4(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_ptr,
                             __global uchar *output, __private const int height,
                             __private const int width) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx         = image_height_idx / height;
    const int height_idx        = image_height_idx % height;

    int buffer_offset = ((batch_idx * height + height_idx) * width + image_width_idx) * 4;
    int2 coord      = (int2)(image_width_idx, image_height_idx);
    float4 values_f = read_imagef(input_ptr, SAMPLER, coord);
    
    uchar4 values = convert_uchar4_sat(values_f);

    vstore4(values, 0, output + buffer_offset);
}
