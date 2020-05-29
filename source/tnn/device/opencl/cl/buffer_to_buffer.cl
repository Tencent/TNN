#include "base.inc"

// convert kernel : from buffer(oihw) to buffer(oc/4 h w ic/4 ic4 oc4)
__kernel void Conv2DFilterBufferToBuffer(GLOBAL_SIZE_2_DIMS __global const float *input_ptr,
                                            __private const int output_channel, __private const int input_channel, __private const int2 kernel_hw, 
                                            __private const int h_w_size, __global FLOAT* output_ptr) {
    int oc_idx = get_global_id(0); // oc
    int ic_h_w_idx = get_global_id(1); // ic h w

    DEAL_NON_UNIFORM_DIM2(oc_idx, ic_h_w_idx);

    const int ic_idx  = ic_h_w_idx / h_w_size;
    const int h_w_idx = ic_h_w_idx % h_w_size;
    const int h_idx   = h_w_idx / kernel_hw.y;
    const int w_idx   = h_w_idx % kernel_hw.y;

    const int ic_size = global_size_dim1 / h_w_size;
    float val = 0;
    if (oc_idx < output_channel && ic_idx < input_channel) {
        const int input_offset = oc_idx * input_channel * h_w_size + ic_idx * h_w_size +
            h_idx * kernel_hw.y + w_idx;
        val = input_ptr[input_offset];
    }

    const int ocb_idx = oc_idx >> 2;
    const int icb_idx = ic_idx >> 2;
    const int output_offset = ocb_idx * ic_size * h_w_size * 4 + h_idx * ic_size * kernel_hw.y * 4 + w_idx * ic_size * 4 + icb_idx * 16 + (ic_idx % 4) * 4 + oc_idx % 4;
    output_ptr[output_offset] = (FLOAT)(val);
}

// convert arg as 4 alignment
__kernel void ArgBufferToBuffer(GLOBAL_SIZE_2_DIMS __global const float *input_ptr, __private const int count,
                                __global FLOAT* output_ptr) {
    int global_x_idx = get_global_id(0);
    int global_y_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(global_x_idx, global_y_idx);

    const int buffer_4_offset = global_x_idx << 2;
    const int remain          = count - buffer_4_offset;

    int offset = buffer_4_offset;
    float4 value = 0;
    if (remain >= 4) {
        value = vload4(0, input_ptr + offset);
    } else if (remain == 3) {
        value.x = *(input_ptr + offset);
        offset++;
        value.y = *(input_ptr + offset);
        offset++;
        value.z = *(input_ptr + offset);
    } else if (remain == 2) {
        value.x = *(input_ptr + offset);
        offset++;
        value.y = *(input_ptr + offset);
    } else if (remain == 1) {
        value.x = *(input_ptr + offset);
    }

    FLOAT4 value_out = (FLOAT4)((FLOAT)(value.x), (FLOAT)(value.y), (FLOAT)(value.z), (FLOAT)(value.w));
    vstore4(value_out, 0, output_ptr + buffer_4_offset);
}
