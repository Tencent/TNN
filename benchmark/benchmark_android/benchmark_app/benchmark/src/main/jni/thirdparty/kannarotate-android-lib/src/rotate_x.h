#ifndef ROTATE_X_H
#define ROTATE_X_H

#ifdef __cplusplus
extern "C" {
#endif

// 1-channel
void rotate_2_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_3_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_4_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_5_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_6_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_7_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_8_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst);

// 2-channel
void rotate_2_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_3_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_4_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_5_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_6_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_7_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_8_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst);

// 3-channel
void rotate_2_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_3_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_4_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_5_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_6_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_7_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_8_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst);

// 4-channel
void rotate_2_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_3_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_4_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_5_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_6_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_7_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst);
void rotate_8_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst);

#ifdef __cplusplus
}
#endif

#endif // ROTATE_X_H
