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

#include "opengl_direct_mem_adapter.h"

#if defined(SHARING_MEM_WITH_OPENGL) && (CL_HPP_TARGET_OPENCL_VERSION >= 120)
namespace TNN_NS {

#ifdef _WIN32

    LRESULT CALLBACK WndProc(HWND    hWnd,           // Handle For This Window
                             UINT    uMsg,           // Message For This Window
                             WPARAM  wParam,         // Additional Message Information
                             LPARAM  lParam)         // Additional Message Information
    {
        switch (uMsg)                                   // Check For Windows Messages
        {
            case WM_ACTIVATE:                           // Watch For Window Activate Message
            {
                LOGE("WM_ACTIVE called\n");
                HDC hdc;
                HGLRC hglrc;
                // obtain a device context for the window
                hdc = GetDC(hWnd);

                // set an appropriate pixel format
                PIXELFORMATDESCRIPTOR pfd = {
                    sizeof(PIXELFORMATDESCRIPTOR),
                    1,
                    PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,    //Flags
                    PFD_TYPE_RGBA,        // The kind of framebuffer. RGBA or palette.
                    32,                   // Colordepth of the framebuffer.
                    0, 0, 0, 0, 0, 0,
                    0,
                    0,
                    0,
                    0, 0, 0, 0,
                    24,                   // Number of bits for the depthbuffer
                    8,                    // Number of bits for the stencilbuffer
                    0,                    // Number of Aux buffers in the framebuffer.
                    PFD_MAIN_PLANE,
                    0,
                    0, 0, 0
                };

                int pixel_format;
                pixel_format = ChoosePixelFormat(hdc, &pfd);
                SetPixelFormat(hdc, pixel_format, &pfd);

                if (hglrc = wglCreateContext(hdc)) {
                    // try to make it the thread's current rendering context
                    if (!wglMakeCurrent(hdc, hglrc)) {
                        LOGE("wgl make current context failed\n");
                    }
                }

                return 0;                               // Return To The Message Loop
            }

            case WM_CLOSE:                              // Did We Receive A Close Message?
            {
                LOGE("WM_CLOSE called\n");
                HDC hdc;
                HGLRC hglrc;
                if (hglrc = wglGetCurrentContext()) {
                    hdc = wglGetCurrentDC();

                    wglMakeCurrent(NULL, NULL);

                    // release the device context
                    ReleaseDC(hWnd, hdc);

                    // delete the rendering context
                    wglDeleteContext(hglrc);
                }
                PostQuitMessage(0);                     // Send A Quit Message
                return 0;                               // Jump Back
            }
        }

        // Pass All Unhandled Messages To DefWindowProc
        return DefWindowProc(hWnd, uMsg, wParam, lParam);
    }
#endif

// Create GLES Environment
Status OpenGLDirectMemAdapter::CreateGlEnv() {
    Status status = TNN_OK;
#ifdef _WIN32

    GLuint      PixelFormat;            // Holds The Results After Searching For A Match
    WNDCLASS    wc;                     // Windows Class Structure
    DWORD       dwExStyle;              // Window Extended Style
    DWORD       dwStyle;                // Window Style
    RECT        WindowRect;             // Grabs Rectangle Upper Left / Lower Right Values
    WindowRect.left = (long)0;            // Set Left Value To 0
    WindowRect.right = (long)width_;       // Set Right Value To Requested Width
    WindowRect.top = (long)0;             // Set Top Value To 0
    WindowRect.bottom = (long)height_;     // Set Bottom Value To Requested Height

    hInstance_          = GetModuleHandle(NULL);                // Grab An Instance For Our Window
    wc.style            = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;   // Redraw On Size, And Own DC For Window.
    wc.lpfnWndProc      = (WNDPROC) WndProc;                    // WndProc Handles Messages
    wc.cbClsExtra       = 0;                                    // No Extra Window Data
    wc.cbWndExtra       = 0;                                    // No Extra Window Data
    wc.hInstance        = hInstance_;                           // Set The Instance
    wc.hIcon            = LoadIcon(NULL, IDI_WINLOGO);          // Load The Default Icon
    wc.hCursor          = LoadCursor(NULL, IDC_ARROW);          // Load The Arrow Pointer
    wc.hbrBackground    = NULL;                                 // No Background Required For GL
    wc.lpszMenuName     = NULL;                                 // We Don't Want A Menu
    wc.lpszClassName    = "OpenGL";                             // Set The Class Name

    if (!RegisterClass(&wc)) {
        return Status(TNNERR_COMMON_ERROR, "failed to register the window class");
    }

    dwExStyle=WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;           // Window Extended Style
    dwStyle=WS_OVERLAPPEDWINDOW;                            // Windows Style
    // Create The Window
    if (!(hWnd_=CreateWindowEx( dwExStyle,                          // Extended Style For The Window
                                "OpenGL",                           // Class Name
                                "Window-Tile",                      // Window Title
                                dwStyle |                           // Defined Window Style
                                WS_CLIPSIBLINGS |                   // Required Window Style
                                WS_CLIPCHILDREN,                    // Required Window Style
                                0, 0,                               // Window Position
                                WindowRect.right-WindowRect.left,   // Calculate Window Width
                                WindowRect.bottom-WindowRect.top,   // Calculate Window Height
                                NULL,                               // No Parent Window
                                NULL,                               // No Menu
                                hInstance_,                         // Instance
                                NULL)))                             // Dont Pass Anything To WM_CREATE
    {
        return Status(TNNERR_COMMON_ERROR, "windows create failed");
    }
    hdc_ = GetDC(hWnd_);
    if (!hdc_) {
        return Status(TNNERR_COMMON_ERROR, "create gl device context failed");
    }
    // set an appropriate pixel format
    PIXELFORMATDESCRIPTOR pfd = {
        sizeof(PIXELFORMATDESCRIPTOR),
        1,
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,    //Flags
        PFD_TYPE_RGBA,        // The kind of framebuffer. RGBA or palette.
        32,                   // Colordepth of the framebuffer.
        0, 0, 0, 0, 0, 0,
        0,
        0,
        0,
        0, 0, 0, 0,
        24,                   // Number of bits for the depthbuffer
        8,                    // Number of bits for the stencilbuffer
        0,                    // Number of Aux buffers in the framebuffer.
        PFD_MAIN_PLANE,
        0,
        0, 0, 0
    };

    int pixel_format;
    pixel_format = ChoosePixelFormat(hdc_, &pfd);
    if (!pixel_format) {
        return Status(TNNERR_COMMON_ERROR, "choose suitable pixel format failed");
    }
    if (!SetPixelFormat(hdc_, pixel_format, &pfd)) {
        return Status(TNNERR_COMMON_ERROR, "set pixel format failed");
    }

    hglrc_ = wglCreateContext(hdc_);
    if (!hglrc_) {
        return Status(TNNERR_COMMON_ERROR, "create gl rendering context failed");
    }

    if (!wglMakeCurrent(hdc_, hglrc_)) {
        return Status(TNNERR_COMMON_ERROR, "active gl rendering context failed");
    }
#elif defined(__ANDROID__)
    // EGL config attributes
    const EGLint confAttr[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, // EGL_WINDOW_BIT EGL_PBUFFER_BIT we will create a pixelbuffer surface
        EGL_RED_SIZE,   8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE,  8,
        EGL_ALPHA_SIZE, 8, // if you need the alpha channel
        EGL_DEPTH_SIZE, 8, // if you need the depth buffer
        EGL_STENCIL_SIZE,8,
        EGL_NONE
    };

    // EGL context attributes
    const EGLint ctxAttr[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };

    // surface attributes
    // the surface size is set to the input frame size
    const EGLint surfaceAttr[] = {
        EGL_WIDTH, 1,
        EGL_HEIGHT,1,
        EGL_NONE
    };
    EGLint eglMajVers, eglMinVers;
    EGLint numConfigs;

    do {
        // 1. Get EGLDisplay object
        egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (egl_display_ == EGL_NO_DISPLAY) {
            // Unable to open connection to local windowing system
            return Status(TNNERR_COMMON_ERROR, "CreateGlesEnv Unable to open connection to local windowing system");
        }

        // 2. Initialize EGL method
        if (!eglInitialize(egl_display_, &eglMajVers, &eglMinVers))
        {
            // Unable to initialize EGL. Handle and recover
            return Status(TNNERR_COMMON_ERROR, "CreateGlesEnv Unable to initialize EGL");
        }

        LOGI("CreateGlesEnv EGL init with version %d.%d\n", eglMajVers, eglMinVers);

        // 3. Get EGLConfig object
        if (!eglChooseConfig(egl_display_, confAttr, &egl_config_, 1, &numConfigs))
        {
            return Status(TNNERR_COMMON_ERROR, "CreateGlesEnv some config is wrong");
        }

        // 4. Create EGLSurface
        egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config_, surfaceAttr);
        if (egl_surface_ == EGL_NO_SURFACE)
        {
            switch(eglGetError())
            {
                case EGL_BAD_ALLOC:
                    // Not enough resources available. Handle and recover
                    return Status(TNNERR_COMMON_ERROR, "CreateGlesEnv Not enough resources available");
                    break;
                case EGL_BAD_CONFIG:
                    // Verify that provided EGLConfig is valid
                    return Status(TNNERR_COMMON_ERROR, "CreateGlesEnv provided EGLConfig is invalid");
                    break;
                case EGL_BAD_PARAMETER:
                    // Verify that the EGL_WIDTH and EGL_HEIGHT are
                    // non-negative values
                    return Status(TNNERR_COMMON_ERROR, "CreateGlesEnv provided EGL_WIDTH and EGL_HEIGHT is invalid");
                    break;
                case EGL_BAD_MATCH:
                    // Check window and EGLConfig attributes to determine
                    // compatibility and pbuffer-texture parameters
                    return Status(TNNERR_COMMON_ERROR, "CreateGlesEnv Check window and EGLConfig attributes");
                    break;
            }
        }

        // 5. Create EGLContext
        egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT, ctxAttr);
        if (egl_context_ == EGL_NO_CONTEXT) {
            EGLint error = eglGetError();
            if (error == EGL_BAD_CONFIG) {
                // Handle error and recover
                return Status(TNNERR_COMMON_ERROR, "CreateGlesEnv EGL_BAD_CONFIG");
                break;
            }
        }

        // 6. bind context
        if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
            return Status(TNNERR_COMMON_ERROR, "CreateGlesEnv MakeCurrent failed");
            break;
        }
    } while (false);
#endif

    return status;
}

// Destroy GLES Env
void OpenGLDirectMemAdapter::DestroyGlEnv() {
#ifdef _WIN32
    if (hglrc_ = wglGetCurrentContext()) {
        hdc_ = wglGetCurrentDC();

        wglMakeCurrent(NULL, NULL);

        // release the device context
        ReleaseDC(hWnd_, hdc_);

        // delete the rendering context
        wglDeleteContext(hglrc_);
    }
#elif defined(__ANDROID__)
    if (egl_display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(egl_display_, egl_context_);
        eglDestroySurface(egl_display_, egl_surface_);
        eglReleaseThread();
        eglTerminate(egl_display_);
    }

    egl_display_ = EGL_NO_DISPLAY;
    egl_surface_ = EGL_NO_SURFACE;
    egl_context_ = EGL_NO_CONTEXT;
#endif
}

cl_mem OpenGLDirectMemAdapter::AcquireCLMem() {
    if (gl_tex_ < 1) {
        LOGE("tex is error, mem object may not init!\n");
        return nullptr;
    }
    cl_int cl_error;
    if (sharing_type_ == GL_CL_SHARING) {
        // GL-CL sharing, depends on deviced-share context
        cl_mem_ =
            clCreateFromGLTexture(context_, mem_flags_, GL_TEXTURE_2D, 0, gl_tex_, &cl_error);
        if (cl_error != CL_SUCCESS) {
            LOGE("clCreateFromGLTexture failed! (ERROR CODE: %d)\n", cl_error);
            return nullptr;
        }
        glFinish();
        cl_mem gl_objects[] = {cl_mem_};
        cl_error = clEnqueueAcquireGLObjects(command_queue_, 1, gl_objects, 0, NULL, NULL);
        if (cl_error != CL_SUCCESS) {
            LOGE("clEnqueueAcquireGLObjects failed! (ERROR CODE: %d)\n", cl_error);
            return nullptr;
        }
        cl_image_2d_ = cl_mem_;
    } else if (sharing_type_ == EGL_SHARING && mem_flags_ == CL_MEM_READ_ONLY) {
#ifndef __ANDROID__
        LOGE("Acquire cl mem got unsupported sharing type/mem flag, sharing type: %d, mem_flag: %d\n", sharing_type_, mem_flags_);
#else
        // EGL_SHARING have some problem in HUAWEI when mem flag is WRITE ONLY
        // we use egl sharing only when it is read only
        if (egl_image_ == EGL_NO_IMAGE_KHR) {
            EGLint attribute_list[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE};
            egl_image_ = eglCreateImageKHR(egl_display_, egl_context_, EGL_GL_TEXTURE_2D_KHR,
                                           (EGLClientBuffer) (size_t) gl_tex_, attribute_list);
            if (egl_image_ == EGL_NO_IMAGE_KHR) {
                LOGE("eglCreateImageKHR failed! %d\n", eglGetError());
                return nullptr;
            }
        }
        cl_mem_ =
            clCreateFromEGLImageKHR(context_, egl_display_, egl_image_, mem_flags_, NULL, &cl_error);
        if (cl_error != CL_SUCCESS) {
            LOGE("clCreateFromEGLImageKHR failed! (ERROR CODE: %d)\n", cl_error);
            return nullptr;
        }
        glFinish();
        cl_mem cl_objects[] = {cl_mem_};
        cl_error = clEnqueueAcquireEGLObjectsKHR(command_queue_, 1, cl_objects, 0, NULL, NULL);
        if (cl_error != CL_SUCCESS) {
            LOGE("clEnqueueAcquireEGLObjectsKHR failed! (ERROR CODE: %d)\n", cl_error);
            return nullptr;
        }
        cl_image_2d_ = cl_mem_;
#endif
    } else {
        LOGE("Acquire cl mem got unsupported sharing type/mem flag, sharing type: %d, mem_flag: %d\n", sharing_type_, mem_flags_);
    }
    return cl_mem_;
}

void OpenGLDirectMemAdapter::ReleaseCLMem() {
    cl_int cl_error;
    if (sharing_type_ == GL_CL_SHARING) {
        cl_mem gl_objects[] = {cl_mem_};
        cl_error = clEnqueueReleaseGLObjects(command_queue_, 1, gl_objects, 0, NULL, NULL);
        if (cl_error != CL_SUCCESS) {
            LOGE("clEnqueueReleaseGLObjects failed! (ERROR CODE: %d)\n", cl_error);
            return;
        }
    } else if (sharing_type_ == EGL_SHARING && mem_flags_ == CL_MEM_READ_ONLY) {
#ifdef __ANDROID__
        cl_mem cl_objects[] = {cl_mem_};
        cl_error = clEnqueueReleaseEGLObjectsKHR(command_queue_, 1, cl_objects, 0, NULL, NULL);
        if (cl_error != CL_SUCCESS) {
            LOGE("clEnqueueReleaseEGLObjectsKHR failed! (ERROR CODE: %d)\n", cl_error);
        }
#endif
    }
    return;
}

void OpenGLDirectMemAdapter::UpdateTexture(std::shared_ptr<Mat> input_mat) const {
    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    float *cpu_float_data = new float[width_ * height_ * 4];
    for (int i = 0; i < width_ * height_ * 4; i++) {
        unsigned char *cpu_data = static_cast<unsigned char *>(input_mat->GetData());
        if (input_mat->GetMatType() == N8UC4) {
            cpu_float_data[i] = (float)cpu_data[i];
        } else if (input_mat->GetMatType() == N8UC3) {
            int channel = i % 4;
            int wh_block = i / 4;
            int index = wh_block * 3 + channel;
            cpu_float_data[i] = (float)cpu_data[index];
        }
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width_, height_, 0, GL_RGBA,
                 GL_FLOAT, cpu_float_data);
    delete cpu_float_data;

}

void OpenGLDirectMemAdapter::GetImage2DInfo(const cl::Image& cl_image,
                                            cl_image_format *format,
                                            size_t *width,
                                            size_t *height,
                                            size_t *slice_pitch,
                                            size_t *row_pitch,
                                            size_t *element_size) const {
    cl_image.getImageInfo(CL_IMAGE_FORMAT, format);
    cl_image.getImageInfo(CL_IMAGE_WIDTH, width);
    cl_image.getImageInfo(CL_IMAGE_HEIGHT, height);
    cl_image.getImageInfo(CL_IMAGE_SLICE_PITCH, slice_pitch);
    cl_image.getImageInfo(CL_IMAGE_ROW_PITCH, row_pitch);
    cl_image.getImageInfo(CL_IMAGE_ELEMENT_SIZE, element_size);
}

Status OpenGLDirectMemAdapter::Transform(std::shared_ptr<Mat> input_mat, std::shared_ptr<Mat>& output_mat,
                                         cl::CommandQueue *command_queue, bool cpu_to_gpu) {
    if (cpu_to_gpu) {
        if (input_mat->GetDeviceType() != DEVICE_NAIVE && input_mat->GetDeviceType() != DEVICE_ARM) {
            return Status(TNNERR_COMMON_ERROR, "input mat device not support");
        }
        if (!input_mat) {
            return Status(TNNERR_COMMON_ERROR, "opengl mem adapter input mat is empty");
        }
        size_t width, height;
        if (!output_mat) {
            output_mat.reset(new Mat(DEVICE_OPENCL, N8UC4, input_mat->GetDims()));
        }
        if (output_mat->GetMatType() == N8UC4) {
            cl::Image *image = static_cast<cl::Image *>(output_mat->GetData());
            if (!image) {
                return Status(TNNERR_COMMON_ERROR, "opengl mem adapter image in output mat is empty");
            }
            image->getImageInfo(CL_IMAGE_WIDTH, &width);
            image->getImageInfo(CL_IMAGE_HEIGHT, &height);
        }
        auto ret = InitWithGLTex(0, width, height, 0, command_queue);
        CHECK_TNN_OK(ret);
        UpdateTexture(input_mat);

        cl_mem ocl_mem_in = AcquireCLMem();
        if (ocl_mem_in == nullptr) {
            return Status(TNNERR_COMMON_ERROR, "input acquire mem failed!");
        }
        #if 0
        ret = MatUtils::Copy(*input_mat, *output_mat, command_queue);
        if (ret != TNN_OK) {
            LOGE("MatUtils::Copy err msg: %s\n", ret.description().c_str());
            return ret;
        }
        #else
        output_mat.reset(new Mat(DEVICE_OPENCL, output_mat->GetMatType(), input_mat->GetDims(), (void *)GetImage2DPtr()));
        #endif
        size_t slice_pitch, row_pitch, element_size;
        cl_image_format format;
        GetImage2DInfo(*(cl::Image *)(output_mat->GetData()), &format, &width, &height, &slice_pitch, &row_pitch, &element_size);
        LOGD("GetImage2DInfo: [format, width, height, slice_pitch, row_pitch, element_size]: [(%d, %d), %d, %d, %d, %d, %d]\n",
             format.image_channel_order, format.image_channel_data_type, width, height, slice_pitch, row_pitch, element_size);

        #if 0
        std::shared_ptr<Mat> out_cpu_mat(new Mat(DEVICE_NAIVE, output_mat->GetMatType(), input_mat->GetDims()));
        ret = MatUtils::Copy(*output_mat, *out_cpu_mat, command_queue);
        CHECK_TNN_OK(ret);
        unsigned char* out_cpu_data = (unsigned char *)out_cpu_mat->GetData();
        unsigned char* input_cpu_data = (unsigned char *)input_mat->GetData();
        LOGI("adapter input cpu data: [%d, %d, %d]\n", input_cpu_data[0], input_cpu_data[1], input_cpu_data[2]);
        LOGI("adapter output opencl data: [%d, %d, %d], output mat_type: %d\n", out_cpu_data[0], out_cpu_data[1], out_cpu_data[2], output_mat->GetMatType());
        #endif
    }
    return TNN_OK;
}

Status OpenGLDirectMemAdapter::InitWithGLTex(GLuint tex, size_t width, size_t height,
                                             int mem_flag, void* command_queue) {
    return InitWithGLTex(tex, width, height, mem_flag, (cl_command_queue)command_queue);
}

Status OpenGLDirectMemAdapter::InitWithGLTex(GLuint tex, size_t width, size_t height,
                                             int mem_flag, cl::CommandQueue* command_queue) {
    return InitWithGLTex(tex, width, height, mem_flag, (*command_queue)());
}

Status OpenGLDirectMemAdapter::InitWithGLTex(GLuint tex, size_t width, size_t height,
                                             int mem_flag, cl_command_queue command_queue) {
    if (tex < 1) {
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex);
        CheckGlError("glGenTextures");
    }
    if (width != width_ || height != height_ || command_queue_ != command_queue || tex != gl_tex_) {
        // dims change, should clean up when naive implement
        CleanUp();
        LOGD("dims change, w: %d, h: %d, id: %d, new_w: %d, new_h: %d, new_id: %d\n",
             width_, height_, gl_tex_,
             width, height, tex);
    }
    gl_tex_ = tex;
    width_ = width;
    height_ = height;
    mem_flags_ = mem_flag == 0 ? CL_MEM_READ_ONLY : CL_MEM_WRITE_ONLY;

    LOGI("opengl adapter mem flags: %d\n", mem_flags_);
    if (command_queue_ != command_queue) {
        command_queue_ = command_queue;
        // get context from command queue
        size_t sz;
        cl_int ret =
            clGetCommandQueueInfo(command_queue_, CL_QUEUE_CONTEXT, sizeof(cl_context), &context_, &sz);
        if (ret != CL_SUCCESS) {
            LOGE("clGetCommandQueueInfo get context error!, ret:%d\n", ret);
            return TNN_OK;
        }
        // get device
        cl_device_id device_id;
        ret = clGetCommandQueueInfo(command_queue_, CL_QUEUE_DEVICE, sizeof(cl_device_id),
                                    &device_id, &sz);
        if (ret != CL_SUCCESS) {
            LOGE("clGetCommandQueueInfo get device error!");
            return TNN_OK;
        }
        // get device name
        char device_name[256] = {0};
        ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, &sz);
        if (ret != CL_SUCCESS) {
            LOGE("clGetDeviceInfo get device name error!");
            return TNN_OK;
        }
        // check cl_khr_egl_image support
        std::string str_device_name(device_name);
        LOGI("device name: %s\n", str_device_name.c_str());

        char device_extension[4096] = {0};
        ret = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(device_extension), &device_extension, &sz);
        if (ret != CL_SUCCESS) {
            LOGE("clGetDeviceInfo get device extensions error!, ret: %d\n", ret);
            return TNN_OK;
        }
        std::string str_device_extension(device_extension);
        bool cl_egl_ext = false;
        if (str_device_extension.find("cl_khr_egl_image") != std::string::npos) {
            cl_egl_ext = true;
        }
        // get context properties，to check share context
        cl_context_properties context_prop[256] = {0};
        ret =
            clGetContextInfo(context_, CL_CONTEXT_PROPERTIES, sizeof(context_prop), &context_prop, &sz);
        if (ret != CL_SUCCESS) {
            LOGE("clGetContextInfo get prop name error!");
            return TNN_OK;
        }
        bool share_context_from_prop = false;
        for (size_t i = 0; i < sz; i++) {
            if (context_prop[i] == CL_GL_CONTEXT_KHR) {
#ifdef _WIN32
                if (context_prop[i + 1] == (cl_context_properties)wglGetCurrentContext()) {
#else
                if (context_prop[i + 1] == (cl_context_properties)eglGetCurrentContext()) {
#endif
                    share_context_from_prop = true;
                    break;
                }
            }
        }
        if (share_context_from_prop) {
            sharing_type_ = GL_CL_SHARING;
        } else if (str_device_name.find("Mali") != std::string::npos && cl_egl_ext) {
            sharing_type_ = EGL_SHARING;
        } else {
            sharing_type_ = NAIVE_MEM;
        }

        LOGI("opengl adapter sharing_type_: %d\n", sharing_type_);
#ifdef __ANDROID__
        egl_display_ = eglGetCurrentDisplay();
        egl_context_ = eglGetCurrentContext();
#endif
    }
    return TNN_OK;
}

OpenGLDirectMemAdapter::OpenGLDirectMemAdapter() {
    Status ret = CreateGlEnv();
    if (ret != TNN_OK) {
        LOGE("CreateGlEnv failed err msg: %s\n", ret.description().c_str());
    }
#ifdef _WIN32
    support_share_context_ = CheckShareContextSupport();
#endif
}

void OpenGLDirectMemAdapter::CleanUp() {
    if (sharing_type_ == EGL_SHARING && mem_flags_ == CL_MEM_READ_ONLY) {
#ifdef __ANDROID__
        if (egl_image_ != EGL_NO_IMAGE_KHR) {
            eglDestroyImageKHR(egl_display_, egl_image_);
            egl_image_ = EGL_NO_IMAGE_KHR;
            LOGD("eglDestroyImageKHR result: %d\n", eglGetError());
        }
        if (cl_mem_ != nullptr) {
            clReleaseMemObject(cl_mem_);
            cl_mem_ = nullptr;
        }
#endif
    }
}

OpenGLDirectMemAdapter::~OpenGLDirectMemAdapter() {
    CleanUp();
    DestroyGlEnv();
}

bool OpenGLDirectMemAdapter::IsInit() {
    return gl_tex_ != 0;
}

cl::Image2D *OpenGLDirectMemAdapter::GetImage2DPtr() {
    return &cl_image_2d_;
}

cl_mem OpenGLDirectMemAdapter::GetCLMem() {
    return cl_mem_;
}

void OpenGLDirectMemAdapter::BindFrameBuffer(int textureId) {
    if (frame_buffer_ <= 0) {
        glGenFramebuffers(1, &frame_buffer_);
        CheckGlError("glGenFramebuffers");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
    CheckGlError("glBindFramebuffer");
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, textureId, 0);
    CheckGlError("glFramebufferTexture2D");
}

bool OpenGLDirectMemAdapter::CheckShareContextSupport() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() <= 0) {
        LOGE("OpenCL Platform not found!\n");
        return false;
    }
    LOGD("find %lu platforms\n", platforms.size());
    // search GPU
    bool is_share_context_support = false;
    std::vector<cl::Device> devices;
    for (auto it = platforms.begin(); it != platforms.end(); ++it) {
        std::string platform_name;
        it->getInfo(CL_PLATFORM_NAME, &platform_name);
        it->getDevices(CL_DEVICE_TYPE_GPU, &devices);
        LOGD("platform (%s) has %lu GPUs\n", platform_name.c_str(), devices.size());
        if (devices.size() > 0) {
            std::string device_name = devices[0].getInfo<CL_DEVICE_NAME>();
            std::string device_extension = devices[0].getInfo<CL_DEVICE_EXTENSIONS>();
            if (device_extension.find("cl_khr_gl_sharing") == std::string::npos) {
                LOGE("GPU (%s) not support share context\n", device_name.c_str());
                continue;
            }
            cl::Platform::setDefault(*it);
            cl_platform_id platform;
            clGetPlatformIDs(1, &platform, NULL);
#ifdef _WIN32
            cl_context_properties context_prop[] = {CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
                                                    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 0};
#elif defined(__ANDROID__)
            cl_context_properties context_prop[] = {CL_GL_CONTEXT_KHR, (cl_context_properties)eglGetCurrentContext(),
                                                    CL_EGL_DISPLAY_KHR, (cl_context_properties)eglGetCurrentDisplay(), 0};
#endif
            cl_int ret;
            std::shared_ptr<cl::Context> context_ = std::shared_ptr<cl::Context>(new cl::Context(devices[0],
                                                                                                 context_prop,
                                                                                                 nullptr,
                                                                                                 nullptr,
                                                                                                 &ret));

            if (ret != CL_SUCCESS) {
                LOGE("GPU (%s) not support share context\n", device_name.c_str());
            } else {
                LOGI("GPU (%s) support opencl share context\n", device_name.c_str());
                is_share_context_support = true;
            }
        }
    }

    return is_share_context_support;
}

}
#endif // opengl
