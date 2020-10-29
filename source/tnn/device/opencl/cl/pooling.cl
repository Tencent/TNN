#include "base.inc"

__kernel void Pooling(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                      __private const int2 input_wh, __private const int output_height, __private const int2 pad_wh,
                      __private const int2 stride_wh,
                      __private const int2 kernel_wh,
                      __write_only image2d_t output) {
    const int output_channel_idx      = get_global_id(0);
    const int output_width_idx        = get_global_id(1);
    const int output_batch_height_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_idx, output_width_idx, output_batch_height_idx);
    const int output_width = global_size_dim1;

    const int output_batch_idx    = output_batch_height_idx / output_height;
    const int output_height_idx   = output_batch_height_idx - mul24(output_batch_idx, output_height);
    const int input_start         = mul24(output_batch_idx, input_wh.y);
    const int input_height_start  = mad24(output_height_idx, stride_wh.y, -pad_wh.y);
    const int input_width_start   = mad24(output_width_idx, stride_wh.x, -pad_wh.x);
    const int input_channel_start = mul24(output_channel_idx, input_wh.x);

#ifdef POOL_AVG
    float4 output_result = 0;
    for (int height = 0; height < kernel_wh.y; height++) {
        int input_height_idx = input_height_start + height;
        input_height_idx =
            select(input_start + input_height_idx, -1, (input_height_idx < 0 || input_height_idx >= input_wh.y));
        for (int width = 0; width < kernel_wh.x; width++) {
            int input_width_idx = input_width_start + width;
            input_width_idx =
                select(input_channel_start + input_width_idx, -1, (input_width_idx < 0 || input_width_idx >= input_wh.x));

            float4 input_data = read_imagef(input, SAMPLER, (int2)(input_width_idx, input_height_idx));
            output_result     = output_result + input_data;
        }
    }

    const int kernel_height_start = max(0, input_height_start);
    const int kernel_width_start  = max(0, input_width_start);
    const int kernel_height_end   = min(input_height_start + kernel_wh.y, input_wh.y);
    const int kernel_width_end    = min(input_width_start + kernel_wh.x, input_wh.x);
    const int block_size = mul24((kernel_height_end - kernel_height_start), (kernel_width_end - kernel_width_start));
    output_result = output_result / (float)block_size;

    const int output_channel_width_idx = mad24(output_channel_idx, output_width, output_width_idx);
    write_imagef(output, (int2)(output_channel_width_idx, output_batch_height_idx), output_result);
#else
    FLOAT4 output_result = (FLOAT4)(-FLT_MAX);
    for (int height = 0; height < kernel_wh.y; height++) {
        int input_height_idx = input_height_start + height;
        input_height_idx =
            select(input_start + input_height_idx, -1, (input_height_idx < 0 || input_height_idx >= input_wh.y));
        if (input_height_idx != -1) {
            for (int width = 0; width < kernel_wh.x; width++) {
                int input_width_idx = input_width_start + width;
                input_width_idx     = select(input_channel_start + input_width_idx, -1,
                                         (input_width_idx < 0 || input_width_idx >= input_wh.x));

                if (input_width_idx != -1) {
                    FLOAT4 input_data = RI_F(input, SAMPLER, (int2)(input_width_idx, input_height_idx));
                    output_result         = fmax(output_result, input_data);
                }
            }
        }
    }

    const int output_channel_width_idx = mad24(output_channel_idx, output_width, output_width_idx);
    WI_F(output, (int2)(output_channel_width_idx, output_batch_height_idx), output_result);
#endif
}
