#include "debug82.h"
#include <arm_neon.h>

void debug82() {
    // // __fp16 x = 1.0f;
    // // __fp16 y = 2.3f;
    // // printf("%f\n", x+y);
    // __fp16 A[8];
    // __fp16 B[8];
    // __fp16 C[8];

    // for (int i = 0; i < 8; i++) {
    //     A[i] = 2.0f;
    //     B[i] = 3.0f;
    //     C[i] = 0.0f;
    // }

    // // asm volatile(
    // //     "vld1.16 {q0}, [%0]\n\t"
    // //     "vld1.16 {q1}, [%1]\n\t"
    // //     "vld1.16 {q2}, [%2]\n\t"
    // //     "vmla.f16 q2, q0, q1\n\t"
    // //     "vst1.16 {q2}, [%2]\n\t"
    // //     :
    // //     :"r"(A),"r"(B),"r"(C)
    // //     :"cc","memory","q0","q1","q2"
    // // );
    // float16x8_t a = vld1q_f16(A);
    // float16x8_t b = vld1q_f16(B);
    // vst1q_f16(C, vmulq_f16(a, b));

    // for (int i = 0; i < 8; i++) {
    //     printf("%f\n", C[i]);
    // }
    printf("from arm82!!!!!!!!!!!!!!!!!");
}