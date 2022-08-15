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

#ifdef TNN_USE_OPENCL_WRAPPER

#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#include <libloaderapi.h>
#else
#include <dlfcn.h>
#endif
#include <memory>
#include <string>
#include <vector>
#include "tnn/device/opencl/opencl_runtime.h"

namespace TNN_NS {
static const std::vector<std::string> g_opencl_library_paths = {
//default opencl library path
#if defined(__APPLE__) || defined(__MACOSX)
    "libOpenCL.so", "/System/Library/Frameworks/OpenCL.framework/OpenCL"
#elif defined(__ANDROID__)
    "libOpenCL.so",
    "libGLES_mali.so",
    "libmali.so",
#if defined(__aarch64__)
    // Qualcomm Adreno
    "/system/vendor/lib64/libOpenCL.so",
    "/system/lib64/libOpenCL.so",
    // Mali
    "/system/vendor/lib64/egl/libGLES_mali.so",
    "/system/lib64/egl/libGLES_mali.so",
    // Pixel Phone
    "libOpenCL-pixel.so",
#else
    // Qualcomm Adreno
    "/system/vendor/lib/libOpenCL.so", "/system/lib/libOpenCL.so",
    // Mali
    "/system/vendor/lib/egl/libGLES_mali.so", "/system/lib/egl/libGLES_mali.so",
    // other
    "/system/vendor/lib/libPVROCL.so", "/data/data/org.pocl.libs/files/lib/libpocl.so",
    // Pixel Phone
    "libOpenCL-pixel.so",
#endif
#elif defined(__linux__)
    "/usr/lib/libOpenCL.so",
    "/usr/local/lib/libOpenCL.so",
    "/usr/local/lib/libpocl.so",
    "/usr/lib64/libOpenCL.so",
    "/usr/lib32/libOpenCL.so",
    "libOpenCL.so"
#elif defined(_WIN32)
    // SysWOW64/OpenCL.dll is 32-bit 
    "C:/Windows/SysWOW64/OpenCL.dll",
    "C:/Windows/System32/OpenCL.dll"
#elif defined(_WIN64)
    "C:/Windows/System32/OpenCL.dll",
    "C:/Windows/SysWOW64/OpenCL.dll"
#endif
};

std::shared_ptr<OpenCLSymbols> OpenCLSymbols::opencl_symbols_singleton_ = nullptr;

OpenCLSymbols *OpenCLSymbols::GetInstance() {
    static std::once_flag opencl_symbol_once;
    std::call_once(opencl_symbol_once, []() { opencl_symbols_singleton_.reset(new OpenCLSymbols()); });

    return opencl_symbols_singleton_.get();
}

OpenCLSymbols::OpenCLSymbols() {
    LOGD("OpenCLSymbols()\n");
}

OpenCLSymbols::~OpenCLSymbols() {
    LOGD("~OpenCLSymbols() start\n");
    if (nullptr == opencl_symbols_singleton_.get())
        return;
    opencl_symbols_singleton_->UnLoadOpenCLLibrary();
    LOGD("~OpenCLSymbols() end\n");
}

//load default library path
bool OpenCLSymbols::LoadOpenCLLibrary() {
    if (handle_ != nullptr) {
        return true;
    }
    for (const auto &opencl_lib : g_opencl_library_paths) {
        if (LoadLibraryFromPath(opencl_lib)) {
            LOGD("OpenCL Lib Path: %s\n", opencl_lib.c_str());
            return true;
        }
    }
    return false;
}

bool OpenCLSymbols::UnLoadOpenCLLibrary() {
    if (handle_ != nullptr) {
#ifdef WIN32
        if (FreeLibrary(handle_) == 0) {
#else
        if (dlclose(handle_) != 0) {
#endif
            return false;
        }
        handle_ = nullptr;
        return true;
    }
    return true;
}

bool OpenCLSymbols::LoadLibraryFromPath(const std::string &library_path) {
#ifdef WIN32
    handle_ = LoadLibraryA(library_path.c_str());
    if (handle_ == nullptr) {
        return false;
    }

#define TNN_LOAD_FUNCTION_PTR(func_name)                                                                               \
    func_name = reinterpret_cast<func_name##Func>(GetProcAddress(handle_, #func_name));                                         \
    if (func_name == nullptr) {                                                                                        \
        LOGE("load func (%s) from (%s) failed!\n", #func_name, library_path.c_str());                                  \
        return false;                                                                                                  \
    }

#else  // WIN32
    handle_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
        return false;
    }
    bool is_pixel = library_path == "libOpenCL-pixel.so";
    typedef void* (*loadOpenCLPointer_t)(const char* name);
    loadOpenCLPointer_t loadOpenCLPointer;
    if(is_pixel){
        typedef void (*enableOpenCL_t)();
        enableOpenCL_t enableOpenCL = reinterpret_cast<enableOpenCL_t>(dlsym(handle_, "enableOpenCL"));
        if (enableOpenCL == nullptr) {
            return false;
        }
        enableOpenCL();
        loadOpenCLPointer = reinterpret_cast<loadOpenCLPointer_t>(dlsym(handle_, "loadOpenCLPointer"));
        if (loadOpenCLPointer == nullptr) {
            return false;
        }
    }

// load function ptr use dlopen and dlsym. if cann't find func_name, will return false.
#define TNN_LOAD_FUNCTION_PTR(func_name)                                                                               \
    if(is_pixel){                                                                                                      \
        func_name = reinterpret_cast<func_name##Func>(loadOpenCLPointer(#func_name));                                  \
    } else {                                                                                                           \
        func_name = reinterpret_cast<func_name##Func>(dlsym(handle_, #func_name));                                     \
    }                                                                                                                  \
    if (func_name == nullptr) {                                                                                        \
        LOGE("load func (%s) from (%s) failed!\n", #func_name, library_path.c_str());                                  \
        return false;                                                                                                  \
    }

#endif // end of WIN32

    TNN_LOAD_FUNCTION_PTR(clGetPlatformIDs);
    TNN_LOAD_FUNCTION_PTR(clGetPlatformInfo);
    TNN_LOAD_FUNCTION_PTR(clBuildProgram);
    TNN_LOAD_FUNCTION_PTR(clEnqueueNDRangeKernel);
    TNN_LOAD_FUNCTION_PTR(clSetKernelArg);
    TNN_LOAD_FUNCTION_PTR(clReleaseKernel);
    TNN_LOAD_FUNCTION_PTR(clCreateProgramWithSource);
    TNN_LOAD_FUNCTION_PTR(clCreateBuffer);
    TNN_LOAD_FUNCTION_PTR(clCreateImage2D);
    TNN_LOAD_FUNCTION_PTR(clCreateImage3D);
    TNN_LOAD_FUNCTION_PTR(clRetainKernel);
    TNN_LOAD_FUNCTION_PTR(clCreateKernel);
    TNN_LOAD_FUNCTION_PTR(clGetProgramInfo);
    TNN_LOAD_FUNCTION_PTR(clFlush);
    TNN_LOAD_FUNCTION_PTR(clFinish);
    TNN_LOAD_FUNCTION_PTR(clReleaseProgram);
    TNN_LOAD_FUNCTION_PTR(clRetainContext);
    TNN_LOAD_FUNCTION_PTR(clGetContextInfo);
    TNN_LOAD_FUNCTION_PTR(clCreateProgramWithBinary);
    TNN_LOAD_FUNCTION_PTR(clCreateCommandQueue);
    TNN_LOAD_FUNCTION_PTR(clGetCommandQueueInfo);
    TNN_LOAD_FUNCTION_PTR(clReleaseCommandQueue);
    TNN_LOAD_FUNCTION_PTR(clEnqueueMapBuffer);
    TNN_LOAD_FUNCTION_PTR(clEnqueueMapImage);
    TNN_LOAD_FUNCTION_PTR(clRetainProgram);
    TNN_LOAD_FUNCTION_PTR(clGetProgramBuildInfo);
    TNN_LOAD_FUNCTION_PTR(clEnqueueReadBuffer);
    TNN_LOAD_FUNCTION_PTR(clEnqueueWriteBuffer);
    TNN_LOAD_FUNCTION_PTR(clWaitForEvents);
    TNN_LOAD_FUNCTION_PTR(clReleaseEvent);
    TNN_LOAD_FUNCTION_PTR(clCreateContext);
    TNN_LOAD_FUNCTION_PTR(clCreateContextFromType);
    TNN_LOAD_FUNCTION_PTR(clReleaseContext);
    TNN_LOAD_FUNCTION_PTR(clRetainCommandQueue);
    TNN_LOAD_FUNCTION_PTR(clEnqueueUnmapMemObject);
    TNN_LOAD_FUNCTION_PTR(clRetainMemObject);
    TNN_LOAD_FUNCTION_PTR(clReleaseMemObject);
    TNN_LOAD_FUNCTION_PTR(clGetDeviceInfo);
    TNN_LOAD_FUNCTION_PTR(clGetDeviceIDs);
    TNN_LOAD_FUNCTION_PTR(clRetainEvent);
    TNN_LOAD_FUNCTION_PTR(clGetKernelWorkGroupInfo);
    TNN_LOAD_FUNCTION_PTR(clGetEventInfo);
    TNN_LOAD_FUNCTION_PTR(clGetEventProfilingInfo);
    TNN_LOAD_FUNCTION_PTR(clGetImageInfo);
    TNN_LOAD_FUNCTION_PTR(clEnqueueCopyImage);
    TNN_LOAD_FUNCTION_PTR(clEnqueueCopyBufferToImage);
    TNN_LOAD_FUNCTION_PTR(clEnqueueCopyImageToBuffer);
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    TNN_LOAD_FUNCTION_PTR(clRetainDevice);
    TNN_LOAD_FUNCTION_PTR(clReleaseDevice);
    TNN_LOAD_FUNCTION_PTR(clCreateImage);
#endif
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    TNN_LOAD_FUNCTION_PTR(clGetKernelSubGroupInfoKHR);
    TNN_LOAD_FUNCTION_PTR(clCreateCommandQueueWithProperties);
    TNN_LOAD_FUNCTION_PTR(clGetExtensionFunctionAddress);
#endif

#undef TNN_LOAD_FUNCTION_PTR

    return true;
}

}  // namespace TNN_NS

// clGetPlatformIDs wrapper, use OpenCLSymbols function. use OpenCLSymbols function.
cl_int CL_API_CALL clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetPlatformIDs;
    CHECK_NOTNULL(func);
    return func(num_entries, platforms, num_platforms);
}

//clGetPlatformInfo wrapper, use OpenCLSymbols function. use OpenCLSymbols function.
cl_int CL_API_CALL clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size,
                         void *param_value, size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetPlatformInfo;
    CHECK_NOTNULL(func);
    return func(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

//clGetDeviceIDs wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices,
                      cl_uint *num_devices) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetDeviceIDs;
    CHECK_NOTNULL(func);
    return func(platform, device_type, num_entries, devices, num_devices);
}

//clGetDeviceInfo wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value,
                       size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetDeviceInfo;
    CHECK_NOTNULL(func);
    return func(device, param_name, param_value_size, param_value, param_value_size_ret);
}

//clCreateContext wrapper, use OpenCLSymbols function.
cl_context CL_API_CALL clCreateContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices,
                           void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *), void *user_data,
                           cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateContext;
    CHECK_NOTNULL(func);
    return func(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

//clCreateContextFromType wrapper, use OpenCLSymbols function.
cl_context CL_API_CALL clCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type,
                                   void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                                   void *user_data, cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateContextFromType;
    CHECK_NOTNULL(func);
    return func(properties, device_type, pfn_notify, user_data, errcode_ret);
}

//clRetainContext wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clRetainContext(cl_context context) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clRetainContext;
    CHECK_NOTNULL(func);
    return func(context);
}

//clReleaseContext wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clReleaseContext(cl_context context) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clReleaseContext;
    CHECK_NOTNULL(func);
    return func(context);
}

//clGetContextInfo wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetContextInfo;
    CHECK_NOTNULL(func);
    return func(context, param_name, param_value_size, param_value, param_value_size_ret);
}

//clCreateProgramWithSource wrapper, use OpenCLSymbols function.
cl_program CL_API_CALL clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths,
                                     cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateProgramWithSource;
    CHECK_NOTNULL(func);
    return func(context, count, strings, lengths, errcode_ret);
}

//clCreateProgramWithBinary wrapper, use OpenCLSymbols function.
cl_program CL_API_CALL clCreateProgramWithBinary(cl_context context, cl_uint count, const cl_device_id *device_list,
                                     const size_t *length, const unsigned char **buffer,
                                     cl_int *binary_status, cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateProgramWithBinary;
    CHECK_NOTNULL(func);
    return func(context, count, device_list, length, buffer, binary_status, errcode_ret);
}

//clGetProgramInfo wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetProgramInfo;
    CHECK_NOTNULL(func);
    return func(program, param_name, param_value_size, param_value, param_value_size_ret);
}

//clGetProgramBuildInfo wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
                             size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetProgramBuildInfo;
    CHECK_NOTNULL(func);
    return func(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

//clRetainProgram wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clRetainProgram(cl_program program) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clRetainProgram;
    CHECK_NOTNULL(func);
    return func(program);
}

//clReleaseProgram wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clReleaseProgram(cl_program program) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clReleaseProgram;
    CHECK_NOTNULL(func);
    return func(program);
}

//clBuildProgram wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options,
                      void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clBuildProgram;
    CHECK_NOTNULL(func);
    return func(program, num_devices, device_list, options, pfn_notify, user_data);
}

//clCreateKernel wrapper, use OpenCLSymbols function.
cl_kernel CL_API_CALL clCreateKernel(cl_program program, const char *kernelName, cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateKernel;
    CHECK_NOTNULL(func);
    return func(program, kernelName, errcode_ret);
}

//clRetainKernel wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clRetainKernel(cl_kernel kernel) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clRetainKernel;
    CHECK_NOTNULL(func);
    return func(kernel);
}

//clReleaseKernel wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clReleaseKernel;
    CHECK_NOTNULL(func);
    return func(kernel);
}

//clSetKernelArg wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clSetKernelArg;
    CHECK_NOTNULL(func);
    return func(kernel, arg_index, arg_size, arg_value);
}

//clCreateBuffer wrapper, use OpenCLSymbols function.
cl_mem CL_API_CALL clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateBuffer;
    CHECK_NOTNULL(func);
    return func(context, flags, size, host_ptr, errcode_ret);
}

//clRetainMemObject wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clRetainMemObject(cl_mem memobj) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clRetainMemObject;
    CHECK_NOTNULL(func);
    return func(memobj);
}

//clReleaseMemObject wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clReleaseMemObject(cl_mem memobj) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clReleaseMemObject;
    CHECK_NOTNULL(func);
    return func(memobj);
}

//clGetImageInfo wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetImageInfo(cl_mem image, cl_image_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetImageInfo;
    CHECK_NOTNULL(func);
    return func(image, param_name, param_value_size, param_value, param_value_size_ret);
}

//clRetainCommandQueue wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clRetainCommandQueue(cl_command_queue command_queue) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clRetainCommandQueue;
    CHECK_NOTNULL(func);
    return func(command_queue);
}

//clReleaseCommandQueue wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue command_queue) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clReleaseCommandQueue;
    CHECK_NOTNULL(func);
    return func(command_queue);
}

//clEnqueueReadBuffer wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset,
                           size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                           cl_event *event) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clEnqueueReadBuffer;
    CHECK_NOTNULL(func);
    return func(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list,
                event);
}

//clEnqueueWriteBuffer wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset,
                            size_t size, const void *ptr, cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list, cl_event *event) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clEnqueueWriteBuffer;
    CHECK_NOTNULL(func);
    return func(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list,
                event);
}

//clEnqueueMapBuffer wrapper, use OpenCLSymbols function.
void *CL_API_CALL clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags,
                         size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                         cl_event *event, cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clEnqueueMapBuffer;
    CHECK_NOTNULL(func);
    return func(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list,
                event, errcode_ret);
}

//clEnqueueMapImage wrapper, use OpenCLSymbols function.
void *CL_API_CALL clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags,
                        const size_t *origin, const size_t *region, size_t *image_row_pitch, size_t *image_slice_pitch,
                        cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event,
                        cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clEnqueueMapImage;
    CHECK_NOTNULL(func);
    return func(command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch, image_slice_pitch,
                num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

//clEnqueueUnmapMemObject wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr,
                               cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clEnqueueUnmapMemObject;
    CHECK_NOTNULL(func);
    return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
}

//clGetKernelWorkGroupInfo wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name,
                                size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetKernelWorkGroupInfo;
    CHECK_NOTNULL(func);
    return func(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

//clGetEventProfilingInfo wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetEventProfilingInfo;
    CHECK_NOTNULL(func);
    return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

//clEnqueueNDRangeKernel wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                              const size_t *global_work_offset, const size_t *global_work_size,
                              const size_t *local_work_size, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clEnqueueNDRangeKernel;
    CHECK_NOTNULL(func);
    return func(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
                num_events_in_wait_list, event_wait_list, event);
}

//clWaitForEvents wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clWaitForEvents;
    CHECK_NOTNULL(func);
    return func(num_events, event_list);
}

//clRetainEvent wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clRetainEvent(cl_event event) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clRetainEvent;
    CHECK_NOTNULL(func);
    return func(event);
}

//clReleaseEvent wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clReleaseEvent(cl_event event) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clReleaseEvent;
    CHECK_NOTNULL(func);
    return func(event);
}

//clGetEventInfo wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetEventInfo;
    CHECK_NOTNULL(func);
    return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

//clFlush wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clFlush(cl_command_queue command_queue) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clFlush;
    CHECK_NOTNULL(func);
    return func(command_queue);
}

//clFinish wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clFinish(cl_command_queue command_queue) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clFinish;
    CHECK_NOTNULL(func);
    return func(command_queue);
}

//clCreateImage2D wrapper, use OpenCLSymbols function.
cl_mem CL_API_CALL clCreateImage2D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, size_t imageWidth,
                       size_t imageHeight, size_t image_row_pitch, void *host_ptr, cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateImage2D;
    CHECK_NOTNULL(func);
    return func(context, flags, image_format, imageWidth, imageHeight, image_row_pitch, host_ptr, errcode_ret);
}

//clCreateImage3D wrapper, use OpenCLSymbols function.
cl_mem CL_API_CALL clCreateImage3D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, size_t imageWidth,
                       size_t imageHeight, size_t imageDepth, size_t image_row_pitch, size_t image_slice_pitch,
                       void *host_ptr, cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateImage3D;
    CHECK_NOTNULL(func);
    return func(context, flags, image_format, imageWidth, imageHeight, imageDepth, image_row_pitch, image_slice_pitch,
                host_ptr, errcode_ret);
}

//clCreateCommandQueue wrapper, use OpenCLSymbols function.
cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties,
                                      cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateCommandQueue;
    CHECK_NOTNULL(func);
    return func(context, device, properties, errcode_ret);
}

//clGetCommandQueueInfo wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size,
                             void *param_value, size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetCommandQueueInfo;
    CHECK_NOTNULL(func);
    return func(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
}

//clEnqueueCopyImage wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clEnqueueCopyImage(cl_command_queue queue, cl_mem src_image, cl_mem dst_image, const size_t *src_origin,
                          const size_t *dst_origin, const size_t *region, cl_uint num_events_in_wait_list,
                          const cl_event *event_wait_list, cl_event *event) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clEnqueueCopyImage;
    CHECK_NOTNULL(func);
    return func(queue, src_image, dst_image, src_origin, dst_origin, region, num_events_in_wait_list, event_wait_list,
                event);
}

//clEnqueueCopyBufferToImage wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image,
                                  size_t src_offset, const size_t *dst_origin, const size_t *region,
                                  cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clEnqueueCopyBufferToImage;
    CHECK_NOTNULL(func);
    return func(command_queue, src_buffer, dst_image, src_offset, dst_origin, region, num_events_in_wait_list,
                event_wait_list, event);
}

//clEnqueueCopyImageToBuffer wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clEnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer,
                                  const size_t *src_origin, const size_t *region, size_t dst_offset,
                                  cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clEnqueueCopyImageToBuffer;
    CHECK_NOTNULL(func);
    return func(command_queue, src_image, dst_buffer, src_origin, region, dst_offset, num_events_in_wait_list,
                event_wait_list, event);
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 120

//clRetainDevice wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clRetainDevice(cl_device_id device) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clRetainDevice;
    CHECK_NOTNULL(func);
    return func(device);
}

//clReleaseDevice wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clReleaseDevice(cl_device_id device) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clReleaseDevice;
    CHECK_NOTNULL(func);
    return func(device);
}

//clCreateImage wrapper, use OpenCLSymbols function.
cl_mem CL_API_CALL clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
                     const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateImage;
    CHECK_NOTNULL(func);
    return func(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200

//clGetKernelSubGroupInfoKHR wrapper, use OpenCLSymbols function.
cl_int CL_API_CALL clGetKernelSubGroupInfoKHR(cl_kernel kernel, cl_device_id device, cl_kernel_sub_group_info param_name,
                                  size_t input_value_size, const void *input_value, size_t param_value_size,
                                  void *param_value, size_t *param_value_size_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetKernelSubGroupInfoKHR;
    CHECK_NOTNULL(func);
    return func(kernel, device, param_name, input_value_size, input_value, param_value_size, param_value,
                param_value_size_ret);
}

//clCreateCommandQueueWithProperties wrapper, use OpenCLSymbols function.
cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(cl_context context, cl_device_id device,
                                                    const cl_queue_properties *properties, cl_int *errcode_ret) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clCreateCommandQueueWithProperties;
    CHECK_NOTNULL(func);
    return func(context, device, properties, errcode_ret);
}

//clGetExtensionFunctionAddress wrapper, use OpenCLSymbols function.
void *CL_API_CALL clGetExtensionFunctionAddress(const char *func_name) {
    auto func = TNN_NS::OpenCLSymbols::GetInstance()->clGetExtensionFunctionAddress;
    CHECK_NOTNULL(func);
    return func(func_name);
}
#endif

#endif  // TNN_USE_OPENCL_WRAPPER
