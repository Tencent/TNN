#include "base.inc"

// the input format:   buffer:(M, K, 1, 1) ==> image:(K/4, M)
// the weight format:  image:(N/4, K)
// the output format:  buffer:(M, N, 1, 1) ==> image:(N/4, M)
__kernel void Innerproduct(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __read_only image2d_t weight, __read_only image2d_t bias,
    __private const int k, __private const int k_remain,
    __write_only image2d_t output) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(col, row);

    FLOAT4 data;
    FLOAT4 weight_0;
    FLOAT4 weight_1;
    FLOAT4 weight_2;
    FLOAT4 weight_3;
    FLOAT4 bias_;
    FLOAT4 sum = (FLOAT4)0;

    int k_size = k;
    if (k_remain > 0) {
        k_size--;
    }

    int y = 0;
    int i = 0;

    for (i = 0; i < k_size; i++) {
        y        = i << 2;
        data     = RI_F(input, SAMPLER, (int2)(i, row));
        weight_0 = RI_F(weight, SAMPLER, (int2)(col, y));
        weight_1 = RI_F(weight, SAMPLER, (int2)(col, y + 1));
        weight_2 = RI_F(weight, SAMPLER, (int2)(col, y + 2));
        weight_3 = RI_F(weight, SAMPLER, (int2)(col, y + 3));

        sum = mad(data.x, weight_0, sum);
        sum = mad(data.y, weight_1, sum);
        sum = mad(data.z, weight_2, sum);
        sum = mad(data.w, weight_3, sum);
    }

    if (k_remain > 0) {
        data = RI_F(input, SAMPLER, (int2)(i, row));
        y    = i << 2;
    }

    switch (k_remain) {
        case 3:
            weight_0 = RI_F(weight, SAMPLER, (int2)(col, y));
            weight_1 = RI_F(weight, SAMPLER, (int2)(col, y + 1));
            weight_2 = RI_F(weight, SAMPLER, (int2)(col, y + 2));
            sum      = mad(data.x, weight_0, sum);
            sum      = mad(data.y, weight_1, sum);
            sum      = mad(data.z, weight_2, sum);
            break;
        case 2:
            weight_0 = RI_F(weight, SAMPLER, (int2)(col, y));
            weight_1 = RI_F(weight, SAMPLER, (int2)(col, y + 1));
            sum      = mad(data.x, weight_0, sum);
            sum      = mad(data.y, weight_1, sum);
            break;
        case 1:
            weight_0 = RI_F(weight, SAMPLER, (int2)(col, y));
            sum      = mad(data.x, weight_0, sum);
            break;
    }
    bias_ = RI_F(bias, SAMPLER, (int2)(col, 0));
    sum += bias_;
    WI_F(output, (int2)(col, row), sum);
}
