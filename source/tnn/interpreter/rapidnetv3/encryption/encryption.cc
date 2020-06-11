// Copyright 2019 Tencent. All Rights Reserved

#include "tnn/interpreter/rapidnetv3/encryption/encryption.h"

#include <stdlib.h>
#include <cassert>
#include <cstring>
#include <memory>

namespace rapidnetv3 {

unsigned char ToHex(unsigned char x)
{
    return  x > 9 ? x + 55 : x + 48;
}
  
unsigned char FromHex(unsigned char x)
{
    unsigned char y;
    if (x >= 'A' && x <= 'Z') y = x - 'A' + 10;
    else if (x >= 'a' && x <= 'z') y = x - 'a' + 10;
    else if (x >= '0' && x <= '9') y = x - '0';
    else assert(0);
    return y;
}
  
std::string UrlEncode(const std::string& str)
{
    std::string strTemp = "";
    size_t length = str.length();
    for (size_t i = 0; i < length; i++)
    {
        if (isalnum((unsigned char)str[i]) ||
            (str[i] == '-') ||
            (str[i] == '_') ||
            (str[i] == '.') ||
            (str[i] == '~'))
            strTemp += str[i];
        else if (str[i] == ' ')
            strTemp += "+";
        else
        {
            strTemp += '%';
            strTemp += ToHex((unsigned char)str[i] >> 4);
            strTemp += ToHex((unsigned char)str[i] % 16);
        }
    }
    return strTemp;
}
  
std::string UrlDecode(const std::string& str)
{
    std::string strTemp = "";
    size_t length = str.length();
    for (size_t i = 0; i < length; i++)
    {
        if (str[i] == '+') strTemp += ' ';
        else if (str[i] == '%')
        {
            assert(i + 2 < length);
            unsigned char high = FromHex((unsigned char)str[++i]);
            unsigned char low = FromHex((unsigned char)str[++i]);
            strTemp += high*16 + low;
        }
        else strTemp += str[i];
    }
    return strTemp;
}


static std::string g_blur_mix_cluster = "u6ch3ZCv@0trNz534f#Fn^FUVADHRWgMLQRTUUYO7574654";

std::shared_ptr<char> BlurMixCluster(std::string cluster)
{
    const uint64_t len = (uint64_t)cluster.length();
    int i = 0, j = 0;
    char* k = new char[256];
    memset(k, 0, 256);
    //extra 1 to insure strlen is 256
    auto mixed_cluster = std::shared_ptr<char>((char*)calloc(256+1, sizeof(char)), [](char* p) {
        free(p);
    });
    auto mixed_cluster_data = mixed_cluster.get();
    
    unsigned char tmp = 0;
    for (i = 0; i<256; i++) {
        mixed_cluster_data[i] = i;
        k[i] = cluster[i%len];
    }
    for (i = 0; i<256; i++)
    {
        j = ((j + mixed_cluster_data[i] + k[i]) % 256 + 256) % 256;
        tmp = mixed_cluster_data[i];
        mixed_cluster_data[i] = mixed_cluster_data[j];
        mixed_cluster_data[j] = tmp;
    }
    
    delete [] k;
    return mixed_cluster;
}

void BlurMixData(const char *data, char *dst, int len)
{
    auto cluster = BlurMixCluster(g_blur_mix_cluster);
    auto cluster_data = cluster.get();
    
    int i = 0, j = 0, t = 0;
    unsigned long k = 0;
    unsigned char tmp;
    for (k = 0; k<len; k++)
    {
        i = (i + 1) % 256;
        j = ((j + cluster_data[i]) % 256 + 256) % 256;
        tmp = cluster_data[i];
        cluster_data[i] = cluster_data[j];//交换s[x]和s[y]
        cluster_data[j] = tmp;
        t = ((cluster_data[i] + cluster_data[j]) % 256 + 256) % 256;
        dst[k] = data[k] ^ cluster_data[t];
    }
}

void BlurMix(const char *data, char *dst, int len)
{
    if (len <= 0) {
        return;
    }
    
    BlurMixData(data, dst, len);
}

std::string BlurMix(const char *data , int len, bool encode)
{
    if (len <= 0) {
        return std::string();
    }
    
    if (encode) {
        //extra 1 to insure strlen is len
        auto mixed_data = std::shared_ptr<char>((char*)calloc(len+1, sizeof(char)), [](char* p) {
            free(p);
        });
        BlurMixData(data, mixed_data.get(), len);
        auto encoded = UrlEncode(std::string(mixed_data.get(), len));
        return encoded;
    } else {
        auto text = UrlDecode(std::string(data, len));
        std::string decoded;
        decoded.resize(text.length(), '\0');
        BlurMixData((const char *)text.c_str(), (char *)decoded.c_str(), (int)text.length());
        return decoded;
    }
}

}  // namespace rapidnetv3
