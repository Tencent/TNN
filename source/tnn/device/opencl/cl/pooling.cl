#include "base.inc"
#include "io.inc"

__kernel void Pooling(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                      __private const int2 input_wh, __private const int output_height, __private const int2 pad_wh,
                      __private const int2 stride_wh,
                      __private const int2 kernel_wh,
                      __write_only image2d_t output) {
    const int output_channel_idx      = get_global_id(0);
    const int output_width_idx        = get_global_id(1);
    const int output_batch_height_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_idx, output_width_idx, output_batch_height_idx);

#ifdef CHECK_INPUT_COOR
    int2 input_dims = get_image_dim(input);
#endif

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

#ifdef CHECK_INPUT_COOR
            int2 input_dims = get_image_dim(input);
            if (!InRange((int2)(input_width_idx, input_height_idx), input_dims)) {
                input_data = (float4)0;
            }
#endif

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

__kernel void PoolingLocal(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                           __private const int2 input_wh, __private const int output_height,
                           __private const int2 pad_wh,
                           __private const int2 stride_wh,
                           __private const int2 kernel_wh,
                           __private const int local_block_size,
                           __private const int2 local_block_size_wh,
                           __private const int2 local_block_count_wh,
                           __write_only image2d_t output,
                           __local FLOAT4* local_output) {
    const int local_id = get_local_id(0);
    const int global_id                 = get_global_id(0);
    const int output_channel_idx        = global_id / local_block_size;
    const int output_width_idx          = get_global_id(1);
    const int output_batch_height_idx   = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(global_id, output_width_idx, output_batch_height_idx);
    const int output_width = global_size_dim1;

    const int output_batch_idx      = output_batch_height_idx / output_height;
    const int output_height_idx     = output_batch_height_idx - mul24(output_batch_idx, output_height);
    const int input_start           = mul24(output_batch_idx, input_wh.y);
    const int input_height_start    = mad24(output_height_idx, stride_wh.y, -pad_wh.y);
    const int input_width_start     = mad24(output_width_idx, stride_wh.x, -pad_wh.x);
    const int input_channel_start   = mul24(output_channel_idx, input_wh.x);
    const int local_width_id        = local_id % local_block_size_wh.x;
    const int local_height_id       = local_id / local_block_size_wh.x;

#ifdef POOL_AVG
    __local float4* avg_output = (__local float4*)local_output;
    avg_output[local_id] = (float4)0;
    int pos_h = local_height_id;

    for (int local_h_block_id = 0; local_h_block_id < local_block_count_wh.y; local_h_block_id++) {
        if (pos_h >= kernel_wh.y) break;
        int pos_w = local_width_id;
        int input_height_idx = input_height_start + pos_h;
        input_height_idx =
            select(input_start + input_height_idx, -1, (input_height_idx < 0 || input_height_idx >= input_wh.y));
        for (int local_w_block_id = 0; local_w_block_id < local_block_count_wh.x; local_w_block_id++) {
            if (pos_w >= kernel_wh.x) break;
            int input_width_idx = input_width_start + pos_w;
            input_width_idx =
                select(input_channel_start + input_width_idx, -1, (input_width_idx < 0 || input_width_idx >= input_wh.x));

            float4 input_data = read_imagef(input, SAMPLER, (int2)(input_width_idx, input_height_idx));
            avg_output[local_id] += input_data;
            pos_w += local_block_size_wh.x;
        }
        pos_h += local_block_size_wh.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride_h = (local_block_size_wh.y >> 1); stride_h > 0; stride_h >>= 1) {
        if (local_height_id < stride_h) {
            avg_output[local_id] += avg_output[local_id + stride_h * local_block_size_wh.x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int stride_w = (local_block_size_wh.x >> 1); stride_w > 0; stride_w >>= 1) {
        if (local_height_id == 0 && local_width_id < stride_w) {
            avg_output[local_id] += avg_output[local_id + stride_w];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        const int kernel_height_start = max(0, input_height_start);
        const int kernel_width_start  = max(0, input_width_start);
        const int kernel_height_end   = min(input_height_start + kernel_wh.y, input_wh.y);
        const int kernel_width_end    = min(input_width_start + kernel_wh.x, input_wh.x);
        const int block_size = mul24((kernel_height_end - kernel_height_start), (kernel_width_end - kernel_width_start));
        avg_output[local_id] = avg_output[local_id] / (float)block_size;

        const int output_channel_width_idx = mad24(output_channel_idx, output_width, output_width_idx);
        write_imagef(output, (int2)(output_channel_width_idx, output_batch_height_idx), avg_output[local_id]);
    }
#else
    local_output[local_id] = (FLOAT4)(-FLT_MAX);
    int pos_h = local_height_id;

    for (int local_h_block_id = 0; local_h_block_id < local_block_count_wh.y; local_h_block_id++) {
        if (pos_h >= kernel_wh.y) break;
        int pos_w = local_width_id;
        int input_height_idx = input_height_start + pos_h;
        input_height_idx =
            select(input_start + input_height_idx, -1, (input_height_idx < 0 || input_height_idx >= input_wh.y));
        if (input_height_idx != -1) {
            for (int local_w_block_id = 0; local_w_block_id < local_block_count_wh.x; local_w_block_id++) {
                if (pos_w >= kernel_wh.x) break;
                int input_width_idx = input_width_start + pos_w;
                input_width_idx =
                    select(input_channel_start + input_width_idx, -1, (input_width_idx < 0 || input_width_idx >= input_wh.x));

                if (input_width_idx != -1) {
                    FLOAT4 input_data = RI_F(input, SAMPLER, (int2)(input_width_idx, input_height_idx));
                    local_output[local_id] = fmax(input_data, local_output[local_id]);
                }
                pos_w += local_block_size_wh.x;
            }
        }
        pos_h += local_block_size_wh.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride_h = (local_block_size_wh.y >> 1); stride_h > 0; stride_h >>= 1) {
        if (local_height_id < stride_h) {
            local_output[local_id] = fmax(local_output[local_id + stride_h * local_block_size_wh.x], local_output[local_id]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int stride_w = (local_block_size_wh.x >> 1); stride_w > 0; stride_w >>= 1) {
        if (local_height_id == 0 && local_width_id < stride_w) {
            local_output[local_id] = fmax(local_output[local_id + stride_w], local_output[local_id]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        const int output_channel_width_idx = mad24(output_channel_idx, output_width, output_width_idx);
        WI_F(output, (int2)(output_channel_width_idx, output_batch_height_idx), local_output[local_id]);
    }
#endif
}
