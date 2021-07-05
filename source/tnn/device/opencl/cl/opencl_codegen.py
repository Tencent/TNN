import os
import sys

def convert_string_to_hex_list(code_str):
    hex_list = []
    for i in range(len(code_str)):
        hex_ = hex(ord(code_str[i]))
        hex_list.append(hex_)
    return hex_list

def opencl_codegen():
    if len(sys.argv) > 1:
        cl_kernel_dir = sys.argv[1]
    else:
        cl_kernel_dir = "./cl"
    output_path = cl_kernel_dir + "/opencl_program.cc"
    if not os.path.exists(cl_kernel_dir):
        print(cl_kernel_dir + " doesn't exist!")

    base_code = ""
    activation_code = ""
    io_code = ""
    file_path = os.path.join(cl_kernel_dir, "base.inc")
    with open(file_path, "r") as f:
        base_code += f.read()

    file_path = os.path.join(cl_kernel_dir, "activation.inc")
    with open(file_path, "r") as f:
        activation_code += f.read()

    file_path = os.path.join(cl_kernel_dir, "io.inc")
    with open(file_path, "r") as f:
        io_code += f.read()

    opencl_code_maps = {}
    for file_name in os.listdir(cl_kernel_dir):
        file_path = os.path.join(cl_kernel_dir, file_name)
        if file_path[-3:] == ".cl":
            with open(file_path, "r") as f:
                code_str = ""
                for line in f.readlines():
                    if "#include \"base.inc\"" in line:
                        code_str += base_code
                    elif "#include \"activation.inc\"" in line:
                        code_str += activation_code
                    elif "#include \"io.inc\"" in line:
                        code_str += io_code
                    else:
                        code_str += line
                opencl_code_maps[file_name[:-3]] = convert_string_to_hex_list(code_str)

    #source model
    opencl_source_map = "#include <map> \n"
    opencl_source_map += "#include <string> \n"
    opencl_source_map += "#include <vector> \n"
    opencl_source_map += "#include \"tnn/core/macro.h\" \n"
    opencl_source_map += "namespace TNN_NS { \n"
    opencl_source_map += "extern const std::map<std::string, std::vector<unsigned char>> g_opencl_program_map = \n{ \n"

    for file_name, file_source in opencl_code_maps.items():
        opencl_source_map += "{\n    \""
        opencl_source_map += file_name
        opencl_source_map += "\", \n"
        opencl_source_map += "    { "
        for source_hex in file_source:
            opencl_source_map += source_hex
            opencl_source_map += ","
        opencl_source_map += "} "
        opencl_source_map += "\n},\n"

    opencl_source_map += "}; \n"
    opencl_source_map += "} \n"

    with open(output_path, "w") as w_file:
        w_file.write(opencl_source_map)

    print("Generate OpenCL Source done !!! \n")

if __name__ == '__main__':
    opencl_codegen()
