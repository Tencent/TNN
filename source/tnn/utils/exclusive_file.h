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

#ifndef TNN_SOURCE_TNN_NETWORK_TENSORRT_EXCLUSIVE_FILE_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORRT_EXCLUSIVE_FILE_H_

#include <stdlib.h>
#include <string>

#if defined _WIN32
#define NOMINMAX
#include <windows.h>
#include <winbase.h>
// Do not remove following statement.
// windows.h replace LoadLibrary with LoadLibraryA, which cause compiling issue of TNN.
#undef LoadLibrary
#elif defined __ANDROID__
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#else
#include <sys/time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#endif

#include "tnn/core/macro.h"

namespace TNN_NS {

const int TNN_NAME_MAX = 128;

typedef struct shared_mutex_struct {
    // Pointer to the pthread mutex and share memory segment.
#if defined _WIN32 
    HANDLE ptr;
#elif defined __ANDROID__
    struct flock* ptr;
#else
    pthread_mutex_t * ptr;
#endif
    int shm_fd;           // Common: Descriptor of shared memory object. Android: File descriptor of the lock.
    char* name;           // Common: Name of the mutex and associated shared memory object. Android: Name of the file lock.
    int created;          // Equals 1 (true) if initialization of this structure caused creation of 
                          // a new shared mutex.
                          // Equals 0 (false) false if this mutex was just retrieved from shared memory. 
} shared_mutex_t;

class ExclFile {
public:
    explicit ExclFile(std::string fname);

    ~ExclFile();

    // @brief try create the file exclusively
    bool Ready();

private:
    bool IsLockFileExists();
    bool IsFileExists();

    std::string m_fname;
    std::string m_lock_name;
    bool m_created;
    shared_mutex_t m_mutex;
};

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORRT_EXCLUSIVE_FILE_H_