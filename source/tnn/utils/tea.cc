// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/utils/tea.h"
#ifdef WIN32
#include <winsock2.h>
#pragma comment(lib,"ws2_32.lib")
#else
#include <netinet/in.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROUNDS 16
#define LOG_ROUNDS 4

typedef uint32_t WORD32;
const WORD32 DELTA = 0x9e3779b9;

static int xrand() {
    static int holdrand = ((rand() % 0xEEEE) << 16) + (unsigned)time(NULL);
    return ((holdrand = holdrand * 214013L + 2531011L) >> 16) & 0x7fff;
}

void TeaEncryptECB(const uint8_t* pInBuf, const uint8_t* pKey, uint8_t* pOutBuf) {
    WORD32 y, z;
    WORD32 sum;
    WORD32 k[4];
    int i;

    y = ntohl(*((WORD32*)pInBuf));
    z = ntohl(*((WORD32*)(pInBuf + 4)));

    for (i = 0; i < 4; i++) {
        k[i] = ntohl(*((WORD32*)(pKey + i * 4)));
    }

    sum = 0;
    for (i = 0; i < ROUNDS; i++) {
        sum += DELTA;
        y += (z << 4) + k[0] ^ z + sum ^ (z >> 5) + k[1];
        z += (y << 4) + k[2] ^ y + sum ^ (y >> 5) + k[3];
    }

    *((WORD32*)pOutBuf)       = htonl(y);
    *((WORD32*)(pOutBuf + 4)) = htonl(z);
}

void TeaDecryptECB(const uint8_t* pInBuf, const uint8_t* pKey, uint8_t* pOutBuf) {
    WORD32 y, z, sum;
    WORD32 k[4];
    int i;

    y = ntohl(*((WORD32*)pInBuf));
    z = ntohl(*((WORD32*)(pInBuf + 4)));

    for (i = 0; i < 4; i++) {
        k[i] = ntohl(*((WORD32*)(pKey + i * 4)));
    }

    sum = DELTA << LOG_ROUNDS;
    for (i = 0; i < ROUNDS; i++) {
        z -= (y << 4) + k[2] ^ y + sum ^ (y >> 5) + k[3];
        y -= (z << 4) + k[0] ^ z + sum ^ (z >> 5) + k[1];
        sum -= DELTA;
    }

    *((WORD32*)pOutBuf)       = htonl(y);
    *((WORD32*)(pOutBuf + 4)) = htonl(z);
}

#define SALT_LEN 2
#define ZERO_LEN 7

void OiSymmetryEncrypt(const uint8_t* pInBuf, int nInBufLen, const uint8_t* pKey, uint8_t* pOutBuf, int* pOutBufLen) {
    int nPadSaltBodyZeroLen;
    int nPadlen;
    uint8_t src_buf[8], zero_iv[8], *iv_buf;
    int src_i, i, j;

    nPadSaltBodyZeroLen = nInBufLen + 1 + SALT_LEN + ZERO_LEN;
    nPadlen             = nPadSaltBodyZeroLen % 8;
    if (nPadlen)
        nPadlen = 8 - nPadlen;

    src_buf[0] = (((uint8_t)xrand()) & 0x0f8) | (uint8_t)nPadlen;
    src_i      = 1;

    while (nPadlen--)
        src_buf[src_i++] = (uint8_t)xrand();

    memset(zero_iv, 0, 8);
    iv_buf      = zero_iv;
    *pOutBufLen = 0;
    for (i = 1; i <= SALT_LEN;) {
        if (src_i < 8) {
            src_buf[src_i++] = (uint8_t)xrand();
            i++;
        }
        if (src_i == 8) {
            for (j = 0; j < 8; j++)
                src_buf[j] ^= iv_buf[j];
            TeaEncryptECB(src_buf, pKey, pOutBuf);
            src_i  = 0;
            iv_buf = pOutBuf;
            *pOutBufLen += 8;
            pOutBuf += 8;
        }
    }

    while (nInBufLen) {
        if (src_i < 8) {
            src_buf[src_i++] = *(pInBuf++);
            nInBufLen--;
        }

        if (src_i == 8) {
            for (i = 0; i < 8; i++)
                src_buf[i] ^= iv_buf[i];
            TeaEncryptECB(src_buf, pKey, pOutBuf);
            src_i  = 0;
            iv_buf = pOutBuf;
            *pOutBufLen += 8;
            pOutBuf += 8;
        }
    }

    for (i = 1; i <= ZERO_LEN;) {
        if (src_i < 8) {
            src_buf[src_i++] = 0;
            i++;  // i inc in here*/
        }

        if (src_i == 8) {
            for (j = 0; j < 8; j++)
                src_buf[j] ^= iv_buf[j];
            TeaEncryptECB(src_buf, pKey, pOutBuf);
            src_i  = 0;
            iv_buf = pOutBuf;
            *pOutBufLen += 8;
            pOutBuf += 8;
        }
    }
}

bool OiSymmetryDecrypt(const uint8_t* pInBuf, int nInBufLen, const uint8_t* pKey, uint8_t* pOutBuf, int* pOutBufLen) {
    int nPadLen, nPlainLen;
    uint8_t dest_buf[8];
    const uint8_t* iv_buf;
    int dest_i, i, j;

    if ((nInBufLen % 8) || (nInBufLen < 16))
        return false;

    TeaDecryptECB(pInBuf, pKey, dest_buf);
    nPadLen = dest_buf[0] & 0x7;
    i       = nInBufLen - 1 - nPadLen - SALT_LEN - ZERO_LEN;
    if (*pOutBufLen < i)
        return false;
    *pOutBufLen = i;
    if (*pOutBufLen < 0)
        return false;

    iv_buf = pInBuf;
    nInBufLen -= 8;
    pInBuf += 8;
    dest_i = 1;
    dest_i += nPadLen;

    for (i = 1; i <= SALT_LEN;) {
        if (dest_i < 8) {
            dest_i++;
            i++;
        }

        if (dest_i == 8) {
            TeaDecryptECB(pInBuf, pKey, dest_buf);
            for (j = 0; j < 8; j++)
                dest_buf[j] ^= iv_buf[j];
            iv_buf = pInBuf;
            nInBufLen -= 8;
            pInBuf += 8;
            dest_i = 0;
        }
    }

    nPlainLen = *pOutBufLen;
    while (nPlainLen) {
        if (dest_i < 8) {
            *(pOutBuf++) = dest_buf[dest_i++];
            nPlainLen--;
        } else if (dest_i == 8) {
            TeaDecryptECB(pInBuf, pKey, dest_buf);
            for (i = 0; i < 8; i++)
                dest_buf[i] ^= iv_buf[i];

            iv_buf = pInBuf;
            nInBufLen -= 8;
            pInBuf += 8;
            dest_i = 0;
        }
    }

    for (i = 1; i <= ZERO_LEN;) {
        if (dest_i < 8) {
            if (dest_buf[dest_i++])
                return false;
            i++;
        } else if (dest_i == 8) {
            TeaDecryptECB(pInBuf, pKey, dest_buf);
            for (j = 0; j < 8; j++)
                dest_buf[j] ^= iv_buf[j];

            iv_buf = pInBuf;
            nInBufLen -= 8;
            pInBuf += 8;
            dest_i = 0;
        }
    }

    return true;
}

int OiSymmetryEncrypt2Len(int nInBufLen) {
    int nPadSaltBodyZeroLen;
    int nPadlen;

    nPadSaltBodyZeroLen = nInBufLen + 1 + SALT_LEN + ZERO_LEN;
    nPadlen             = nPadSaltBodyZeroLen % 8;
    if (nPadlen)
        nPadlen = 8 - nPadlen;

    return nPadSaltBodyZeroLen + nPadlen;
}

static void CheckZeroEncrypt(const uint8_t* pKey, uint8_t* pOutBuf, int* pOutBufLen, uint8_t* src_buf,
                               uint8_t* iv_plain, uint8_t* iv_crypt, int src_i);
static bool CheckZeroDecrypt(const uint8_t* pInBuf, int nInBufLen, const uint8_t* pKey, uint8_t* dest_buf,
                               const uint8_t* iv_pre_crypt, const uint8_t* iv_cur_crypt, int dest_i, int nBufPos);


void OiSymmetryEncrypt2(const uint8_t* pInBuf, int nInBufLen, const uint8_t* pKey, uint8_t* pOutBuf,
                          int* pOutBufLen) {
    int nPadSaltBodyZeroLen;
    int nPadlen;
    uint8_t src_buf[8], iv_plain[8], *iv_crypt;
    int src_i, i, j;

    nPadSaltBodyZeroLen = nInBufLen + 1 + SALT_LEN + ZERO_LEN;
    nPadlen             = nPadSaltBodyZeroLen % 8;
    if (nPadlen)
        nPadlen = 8 - nPadlen;

    src_buf[0] = (((uint8_t)xrand()) & 0x0f8) | (uint8_t)nPadlen;
    src_i      = 1;

    while (nPadlen--)
        src_buf[src_i++] = (uint8_t)xrand();

    for (i = 0; i < 8; i++)
        iv_plain[i] = 0;
    iv_crypt = iv_plain;

    *pOutBufLen = 0;

    for (i = 1; i <= SALT_LEN;) {
        if (src_i < 8) {
            src_buf[src_i++] = (uint8_t)xrand();
            i++;
        }

        if (src_i == 8) {
            for (j = 0; j < 8; j++)
                src_buf[j] ^= iv_crypt[j];

            TeaEncryptECB(src_buf, pKey, pOutBuf);
            for (j = 0; j < 8; j++)
                pOutBuf[j] ^= iv_plain[j];

            for (j = 0; j < 8; j++)
                iv_plain[j] = src_buf[j];

            src_i    = 0;
            iv_crypt = pOutBuf;
            *pOutBufLen += 8;
            pOutBuf += 8;
        }
    }

    while (nInBufLen) {
        if (src_i < 8) {
            src_buf[src_i++] = *(pInBuf++);
            nInBufLen--;
        }

        if (src_i == 8) {
            for (j = 0; j < 8; j++)
                src_buf[j] ^= iv_crypt[j];
            TeaEncryptECB(src_buf, pKey, pOutBuf);

            for (j = 0; j < 8; j++)
                pOutBuf[j] ^= iv_plain[j];

            for (j = 0; j < 8; j++)
                iv_plain[j] = src_buf[j];

            src_i    = 0;
            iv_crypt = pOutBuf;
            *pOutBufLen += 8;
            pOutBuf += 8;
        }
    }

    CheckZeroEncrypt(pKey, pOutBuf, pOutBufLen, src_buf, iv_plain, iv_crypt, src_i);
}

void CheckZeroEncrypt(const uint8_t* pKey, uint8_t* pOutBuf, int* pOutBufLen, uint8_t* src_buf, uint8_t* iv_plain,
                        uint8_t* iv_crypt, int src_i) {
    int i, j;
    for (i = 1; i <= ZERO_LEN;) {
        if (src_i < 8) {
            src_buf[src_i++] = 0;
            i++;
        }

        if (src_i == 8) {
            for (j = 0; j < 8; j++)
                src_buf[j] ^= iv_crypt[j];
            TeaEncryptECB(src_buf, pKey, pOutBuf);

            for (j = 0; j < 8; j++)
                pOutBuf[j] ^= iv_plain[j];

            for (j = 0; j < 8; j++)
                iv_plain[j] = src_buf[j];

            src_i    = 0;
            iv_crypt = pOutBuf;
            *pOutBufLen += 8;
            pOutBuf += 8;
        }
    }
}

bool OiSymmetryDecrypt2(const uint8_t* pInBuf, int nInBufLen, const uint8_t* pKey, uint8_t* pOutBuf,
                          int* pOutBufLen) {
    int nPadLen, nPlainLen;
    uint8_t dest_buf[8], zero_buf[8];
    const uint8_t *iv_pre_crypt, *iv_cur_crypt;
    int dest_i, i, j;
    int nBufPos;
    nBufPos = 0;

    if ((nInBufLen % 8) || (nInBufLen < 16))
        return false;

    TeaDecryptECB(pInBuf, pKey, dest_buf);
    nPadLen = dest_buf[0] & 0x7;
    i       = nInBufLen - 1 - nPadLen - SALT_LEN - ZERO_LEN;
    if ((*pOutBufLen < i) || (i < 0))
        return false;
    *pOutBufLen = i;

    for (i = 0; i < 8; i++)
        zero_buf[i] = 0;

    iv_pre_crypt = zero_buf;
    iv_cur_crypt = pInBuf;
    pInBuf += 8;
    nBufPos += 8;
    dest_i = 1;
    dest_i += nPadLen;

    for (i = 1; i <= SALT_LEN;) {
        if (dest_i < 8) {
            dest_i++;
            i++;
        } else if (dest_i == 8) {
            iv_pre_crypt = iv_cur_crypt;
            iv_cur_crypt = pInBuf;
            for (j = 0; j < 8; j++) {
                if ((nBufPos + j) >= nInBufLen)
                    return false;
                dest_buf[j] ^= pInBuf[j];
            }

            TeaDecryptECB(dest_buf, pKey, dest_buf);
            pInBuf += 8;
            nBufPos += 8;
            dest_i = 0;
        }
    }

    nPlainLen = *pOutBufLen;
    while (nPlainLen) {
        if (dest_i < 8) {
            *(pOutBuf++) = dest_buf[dest_i] ^ iv_pre_crypt[dest_i];
            dest_i++;
            nPlainLen--;
        } else if (dest_i == 8) {
            iv_pre_crypt = iv_cur_crypt;
            iv_cur_crypt = pInBuf;
            for (j = 0; j < 8; j++) {
                if ((nBufPos + j) >= nInBufLen)
                    return false;
                dest_buf[j] ^= pInBuf[j];
            }

            TeaDecryptECB(dest_buf, pKey, dest_buf);
            pInBuf += 8;
            nBufPos += 8;
            dest_i = 0;
        }
    }
    return CheckZeroDecrypt(pInBuf, nInBufLen, pKey, dest_buf, iv_pre_crypt, iv_cur_crypt, dest_i, nBufPos);
}

static bool CheckZeroDecrypt(const uint8_t* pInBuf, int nInBufLen, const uint8_t* pKey, uint8_t* dest_buf,
                               const uint8_t* iv_pre_crypt, const uint8_t* iv_cur_crypt, int dest_i, int nBufPos) {
    int i, j;
    for (i = 1; i <= ZERO_LEN;) {
        if (dest_i < 8) {
            if (dest_buf[dest_i] ^ iv_pre_crypt[dest_i])
                return false;
            dest_i++;
            i++;
        } else if (dest_i == 8) {
            iv_pre_crypt = iv_cur_crypt;
            iv_cur_crypt = pInBuf;
            for (j = 0; j < 8; j++) {
                if ((nBufPos + j) >= nInBufLen)
                    return false;
                dest_buf[j] ^= pInBuf[j];
            }

            TeaDecryptECB(dest_buf, pKey, dest_buf);
            pInBuf += 8;
            nBufPos += 8;
            dest_i = 0;
        }
    }

    return true;
}
//QQ used
int SymmetryEncrypt3Len(int nInBufLen) {
    int nPadSaltBodyZeroLen;
    int nPadlen;

    nPadSaltBodyZeroLen = nInBufLen + 1 + SALT_LEN + ZERO_LEN;
    nPadlen             = nPadSaltBodyZeroLen % 8;
    if (nPadlen)
        nPadlen = 8 - nPadlen;

    return nPadSaltBodyZeroLen + nPadlen;
}

void FourBytesEncryptAFrame(short* v, short* k) {
    short m_n4BytesScheduleDelta = 0x325f;

    short y = v[0], z = v[1], sum = 0, n = 32;
    while (n-- > 0) {
        sum += m_n4BytesScheduleDelta;
        y += (z << 4) + k[0] ^ z + sum ^ (z >> 5) + k[1];
        z += (y << 4) + k[2] ^ y + sum ^ (y >> 5) + k[3];
    }
    v[0] = y;
    v[1] = z;
}

void FourBytesDecryptAFrame(short* v, short* k) {
    short m_n4BytesScheduleDelta = 0x325f;

    short n = 32, sum, y = v[0], z = v[1];
    sum = m_n4BytesScheduleDelta << 5;

    while (n-- > 0) {
        z -= (y << 4) + k[2] ^ y + sum ^ (y >> 5) + k[3];
        y -= (z << 4) + k[0] ^ z + sum ^ (z >> 5) + k[1];
        sum -= m_n4BytesScheduleDelta;
    }
    v[0] = y;
    v[1] = z;
}

int TeaEncrypt(const uint8_t* plain, int plain_len, const uint8_t* key, uint8_t* buf, int buf_len) {
    int pbuflen = buf_len;
    OiSymmetryEncrypt2(plain, plain_len, key, buf, &pbuflen);
    return pbuflen;
}

int TeaDecrypt(const uint8_t* cryptograph, int cryptograph_len, const uint8_t* key, uint8_t* buf, int buf_len) {
    int pbuflen = buf_len;
    if (!OiSymmetryDecrypt2(cryptograph, cryptograph_len, key, buf, &pbuflen)) {
        return -1;
    }
    return pbuflen;
}
