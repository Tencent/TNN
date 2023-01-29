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

#ifndef TNN_EXAMPLES_BASE_OPENGL_DIRECT_MEM_ADAPTER_H_
#define TNN_EXAMPLES_BASE_OPENGL_DIRECT_MEM_ADAPTER_H_

#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include <string>

#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/mat_utils.h"

#if defined(SHARING_MEM_WITH_OPENGL) && (CL_HPP_TARGET_OPENCL_VERSION >= 120)
#include "../../source/tnn/device/opencl/opencl_wrapper.h"
#if _WIN32
#include <windows.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "CL/cl_gl.h"
#else
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>

#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl3ext.h>
#include "CL/cl_egl.h"
#endif

namespace TNN_NS {
/**
 * TNN zero copy mem adapter
 * get direct cl from gl, depend on extention support on device
 * currently, support init with gl, and get cl from it
 */
class OpenGLDirectMemAdapter {
public:
    /**
     * Direct Mem type
     * GL_CL_SHARING: use cl_gl_sharing extention, share mem with gl and cl
     * EGL_SHARING: use egl extension, gl transfer to egl, and then transfer to cl
     */
    enum DirectMemType {
        GL_CL_SHARING,
        EGL_SHARING,
        NAIVE_MEM,
    };
    /**
     * check share context support
     * if share context support, we can use GL_CL_SHARING
     * @return true is support, false if not
     */
    bool CheckShareContextSupport();

    OpenGLDirectMemAdapter();
    /**
     * init a direct mem object with a gl texture
     * @param tex gl texture
     * @param width texture width
     * @param height texture height
     * @param mem_flag mem_falg, 0 read only, 1 write only
     * @return 0 sucess
     */
    Status InitWithGLTex(GLuint tex, size_t width, size_t height, int mem_flag,
                         void* command_queue);

    /**
     * init a direct mem object with a gl texture
     * @param tex gl texture
     * @param width texture width
     * @param height texture height
     * @param mem_flag mem_falg, 0 read only, 1 write only
     * @param cl::CommandQueue cl cpp comman queue
     * @return 0 sucess
     */
    Status InitWithGLTex(GLuint tex, size_t width, size_t height, int mem_flag,
                         cl::CommandQueue *command_queue);
    /**
     * get init state
     * @return true if init
     */
    bool IsInit();
    /**
     * get cl::Image2D ptr
     * @return
     */
    cl_mem GetCLMem();

    /**
     * get cl::Image2D ptr
     */
    cl::Image2D *GetImage2DPtr();

    virtual ~OpenGLDirectMemAdapter();

 private:
    /**
     * Acqire cl mem from a gl texture
     * Must run in a GL thread, cl_mem can be direct map from gl texture,
     * or new create, depend on GPU platform.
     */
    cl_mem AcquireCLMem();

    /**
     * Release cl mem from a gl texture
     */
    void ReleaseCLMem();

    /**
     * we should bind frame buffer before readpixels
     * @param textureId
     */
    void BindFrameBuffer(int textureId);

    /**
     * init a direct mem object with a gl texture
     * @param tex gl texture
     * @param width texture width
     * @param height texture height
     * @param mem_flag mem_falg, 0 read only, 1 write only
     * @param cl_command_queue cl c comman queue
     * @return 0 sucess
     */
    Status InitWithGLTex(GLuint tex, size_t width, size_t height, int mem_flag,
                         cl_command_queue command_queue);

    /**
     * clean internal mem
     */
    void CleanUp();

    /**
     * get image2d info
     */
    void GetImage2DInfo(const cl::Image &cl_image, cl_image_format *format, size_t *width, size_t *height, size_t *slice_pitch,
                        size_t *row_pitch, size_t *element_size) const;

    /**
     * use cpu data update texture
     */
    void UpdateTexture(std::shared_ptr<Mat> input_mat) const;

    Status CreateGlEnv();

    void DestroyGlEnv();

 public:

    /**
     * RetainCLImage cl image2d from a gl texture
     * Must run in a GL thread, cl_mem can be direct map from gl texture,
     * or new create, depend on GPU platform.
     * tnn use this api
     */
    Status RetainCLImage(void *image, cl::CommandQueue* command_queue, int mem_flg);

    /**
     * Transform tnn cpu mat -> gl texture -> tnn opencl mat
     */
    Status Transform(std::shared_ptr<Mat> input_mat, std::shared_ptr<Mat>& output_mat, cl::CommandQueue *command_queue, bool cpu_to_gpu=true);

 private:
    GLuint gl_tex_ = 0;                           // gl texture
    cl_mem cl_mem_ = nullptr;                     // cl mem object
    unsigned char *cpu_data_ = nullptr;           // cpu data address
    size_t width_ = 224;                          // data width
    size_t height_ = 224;                         // data height
    cl_mem_flags mem_flags_;                      // mem flag, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY
    cl_command_queue command_queue_ = nullptr;    // cl command queue
    cl_context context_;                          // cl context
    DirectMemType sharing_type_;                  // sharing type, glcl, egl, naive
#ifdef _WIN32
    HINSTANCE hInstance_;
    HGLRC hglrc_;
    HDC hdc_;
    HWND hWnd_;
#else
    EGLImageKHR egl_image_ = EGL_NO_IMAGE_KHR;    // some platform should use egl to zero copy
    EGLDisplay egl_display_ = nullptr;            // egl display
    EGLContext egl_context_ = nullptr;            // egl context
    EGLConfig egl_config_ = nullptr;              // egl config
    EGLSurface egl_surface_ = nullptr;            // egl surface
#endif
    cl::Image2D cl_image_2d_;                     // from cl_mem
    GLuint frame_buffer_ = 0;                     // frame buffer for gl readpixels
    bool support_share_context_ = false;
};

/**
 * check gl oprator, if thers are errors, print out
 * @param op
 */
static inline void CheckGlError(const char *op) {
    for (GLint error = glGetError(); error; error = glGetError()) {
        LOGE("after %s() glError (0x%x)\n", op, error);
    }
  return;
}

}
#endif // opengl

#endif