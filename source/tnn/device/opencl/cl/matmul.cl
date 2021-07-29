#include "base.inc"

// matrix_a format: image:(K/4, batch_a * M)
// matrix_b format: image:(N/4, batch_b * K)
// matrix_c format: image:(N/4, batch_c * M)
__kernel void MatMul(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t matrix_a,
    __read_only image2d_t matrix_b,
    __private const int m,
    __private const int k_blocks, __private const int k, __private const int k_remain,
    __private const int batch_a, __private const int batch_b,
    __write_only image2d_t matrix_c) {
    const int image_row = get_global_id(0);
    const int image_col = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_row, image_col);

    int batch_c_idx = image_col / m;
    int m_idx       = image_col % m;
    int batch_a_idx = select(0, batch_c_idx, batch_c_idx < batch_a);
    int batch_b_idx = select(0, batch_c_idx, batch_c_idx < batch_b);

    FLOAT4 matrix_a_data;
    FLOAT4 matrix_b_data_0;
    FLOAT4 matrix_b_data_1;
    FLOAT4 matrix_b_data_2;
    FLOAT4 matrix_b_data_3;
    FLOAT4 sum = (FLOAT4)0;

    int k_size = k_blocks;
    if (k_remain > 0) {
        k_size--;
    }

    int matrix_a_y_idx = batch_a_idx * m + m_idx;
    int matrix_b_y_offset = batch_b_idx * k;
    int y = matrix_b_y_offset, i = 0;
    for (; i < k_size; i++) {
        matrix_a_data = RI_F(matrix_a, SAMPLER, (int2)(i, matrix_a_y_idx));
        matrix_b_data_0 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y));
        matrix_b_data_1 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 1));
        matrix_b_data_2 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 2));
        matrix_b_data_3 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 3));

        sum = mad(matrix_a_data.x, matrix_b_data_0, sum);
        sum = mad(matrix_a_data.y, matrix_b_data_1, sum);
        sum = mad(matrix_a_data.z, matrix_b_data_2, sum);
        sum = mad(matrix_a_data.w, matrix_b_data_3, sum);
        y += 4;
    }

    if (k_remain > 0) {
        matrix_a_data = RI_F(matrix_a, SAMPLER, (int2)(i, matrix_a_y_idx));
    }

    switch (k_remain) {
        case 3:
            matrix_b_data_0 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y));
            matrix_b_data_1 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 1));
            matrix_b_data_2 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 2));
            sum      = mad(matrix_a_data.x, matrix_b_data_0, sum);
            sum      = mad(matrix_a_data.y, matrix_b_data_1, sum);
            sum      = mad(matrix_a_data.z, matrix_b_data_2, sum);
            break;
        case 2:
            matrix_b_data_0 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y));
            matrix_b_data_1 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 1));
            sum      = mad(matrix_a_data.x, matrix_b_data_0, sum);
            sum      = mad(matrix_a_data.y, matrix_b_data_1, sum);
            break;
        case 1:
            matrix_b_data_0 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y));
            sum      = mad(matrix_a_data.x, matrix_b_data_0, sum);
            break;
    }
    WI_F(matrix_c, (int2)(image_row, image_col), sum);
}
