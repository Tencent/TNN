#include <cstddef>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cfloat>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>


static void SanitizeName(char* name)
{
    for (std::size_t i = 0; i < strlen(name); i++)
    {
        if (!isalnum(name[i]))
        {
            name[i] = '_';
        }
    }
}

static std::string PathtoVarname(const char* path)
{
    const char* lastslash = strrchr(path, '/');
    const char* name = lastslash == NULL ? path : lastslash + 1;

    std::string varname = name;
    SanitizeName((char*)varname.c_str());

    return varname;
}

static int DumpProto(const char* protopath, const char* modelpath, const char* idcpppath)
{
    FILE* fp = fopen(protopath, "rb");
    FILE* mp = fopen(modelpath, "rb");

    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", protopath);
        return -1;
    }

    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", modelpath);
        return -1;
    }
    std::string proto_var = PathtoVarname(protopath);
    std::string model_var = PathtoVarname(modelpath);
    std::string include_guard_var = PathtoVarname(idcpppath);

    FILE* ip = fopen(idcpppath, "wb");

    fprintf(ip, "#ifndef TNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());
    fprintf(ip, "#define TNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

    fprintf(ip, "\n#ifdef _MSC_VER\n__declspec(align(4))\n#else\n__attribute__((aligned(4)))\n#endif\n");

    fprintf(ip, "static const unsigned char %s[] = {\n", proto_var.c_str());
    int i = 0;
    int j = 0;
    int c;
 
    while(1)
    {
        c = fgetc(fp);
        if( feof(fp) )
        { 
            break ;
        }
        fprintf(ip, "0x%02x,", c);
        j++;
        if (j % 16 == 0)
        {
            fprintf(ip, "\n");
        }
    }
    fprintf(ip, "};\n");


    std::ifstream model_stream(modelpath);
    std::string model_content =
                    std::string((std::istreambuf_iterator<char>(model_stream)), std::istreambuf_iterator<char>());


    fprintf(ip, "static const unsigned char %s[] = {\n", model_var.c_str());


    while(1)
    {
        c = fgetc(mp);
        if( feof(mp) )
        { 
            break ;
        }
        fprintf(ip, "0x%02x,", c);
        i++;
        if (i % 16 == 0)
        {
            fprintf(ip, "\n");
        }
    }
    fprintf(ip, "};\n");

    fprintf(ip, "static const int %s_longth = {\n", model_var.c_str());
    fprintf(ip, "%u", i);
    fprintf(ip, "};\n");

    fprintf(ip, "static const int %s_longth = {\n", proto_var.c_str());
    fprintf(ip, "%u", j);
    fprintf(ip, "};\n");

    fprintf(ip, "#endif // TNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

    fclose(fp);
    fclose(mp);
    fclose(ip);
    return 0;
}


int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [tnnproto] [tnnmodel] [memcpppath]\n", argv[0]);
        return -1;
    }

    const char* protopath = argv[1];
    const char* modelpath = argv[2];
    const char* memcpppath = argv[3];
    DumpProto(protopath, modelpath, memcpppath);
    return 0;
}
