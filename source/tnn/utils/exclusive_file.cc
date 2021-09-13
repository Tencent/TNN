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

#include "tnn/utils/exclusive_file.h"

#include <string.h>
#include <stdio.h>
#include <mutex>

#if defined _WIN32 
#include <windows.h>
#include <winbase.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#include "tnn/core/macro.h"
#include "tnn/utils/md5.h"

namespace TNN_NS {

static std::mutex static_mutex;

#if defined(_WIN32)

shared_mutex_t shared_mutex_init(const char* iname) {
    shared_mutex_t mutex = {NULL, 0, NULL, 1};
    std::string iname_md5 = md5(std::string(iname));

    mutex.name = (char*)malloc(TNN_NAME_MAX + 1);
    sprintf(mutex.name, ".%s.tnnmutex", iname_md5.c_str());

    mutex.ptr = CreateMutex( 
        NULL,              // default security attributes
        FALSE,             // initially not owned
        mutex.name);             

    if (mutex.ptr == NULL) 
    {
        printf("CreateMutex error: %d\n", GetLastError());
        exit(-1);
    }

    return mutex;
}

int shared_mutex_close(shared_mutex_t m) {
    if (m.ptr != NULL) {
        if (!CloseHandle(m.ptr)) {
           printf("CloseHandle failed\n");
           return -1;
        }
        m.ptr = NULL;
    }
    free(m.name);
    return 0;
}

void shared_mutex_lock(shared_mutex_t * mutex) {
    // no time-out interval
    DWORD dwWaitResult = WaitForSingleObject(mutex->ptr, INFINITE);     
    switch (dwWaitResult) 
    {
            // The thread got ownership of the mutex
            case WAIT_OBJECT_0: 
                break; 

            // The thread got ownership of an abandoned mutex
            // Since windows Mutex has no api to fix the mutex now,
            // we exit(here) to avoid further undefined behavior.
            case WAIT_ABANDONED: 
                return; 
    }
}

void shared_mutex_unlock(shared_mutex_t * mutex) {
    if (! ReleaseMutex(mutex->ptr)) 
    { 
        // Handle error.
    } 
}

static bool file_exists(const char * fname) {
    auto ret = GetFileAttributes(fname); 
    if(INVALID_FILE_ATTRIBUTES == ret && GetLastError() == ERROR_FILE_NOT_FOUND)
    {
        return false;
    }
    return true;
}

static void create_file(const char * fname) {
    HANDLE h = CreateFile(fname,    // name of the file
                          GENERIC_WRITE, // open for writing
                          0,             // sharing mode, none in this case
                          0,             // use default security descriptor
                          CREATE_NEW, // overwrite if exists
                          FILE_ATTRIBUTE_NORMAL,
                          0);
    if (h)
    {
        CloseHandle(h);
    }
}

#elif defined(__ANDROID__) // Android

shared_mutex_t shared_mutex_init(const char* iname) {
    shared_mutex_t mutex = {NULL, 0, NULL, 1};
    std::string iname_str = std::string(iname);
    std::string iname_md5 = md5(iname_str);
    
    size_t pos = iname_str.find_last_of("\\/");
    std::string idir = (std::string::npos == pos) ? "" : iname_str.substr(0, pos);

    mutex.name = (char*)malloc(TNN_NAME_MAX + 1);
    sprintf(mutex.name, "%s/.%u.%s.tnnmutex", idir.c_str(), getuid(), iname_md5.c_str());

    // @brief open existing lock file, or create one.
    mutex.shm_fd = open(mutex.name, O_RDWR | O_CREAT, 0660);
    if (mutex.shm_fd < 0) {
        perror("lock file of mutex open failed");
        return mutex;
    }

    mutex.ptr = (struct flock*)malloc(sizeof(struct flock));
    mutex.ptr->l_whence = SEEK_SET;
    mutex.ptr->l_start = 0;
    mutex.ptr->l_len = 0;
    mutex.ptr->l_pid = 0;

    return mutex;
}

int shared_mutex_close(shared_mutex_t mutex) {
    if (mutex.ptr != NULL) {
        free(mutex.ptr);
        mutex.ptr = NULL;
    }
    if (mutex.shm_fd >= 0 && close(mutex.shm_fd)) {
        perror("lock file of mutex close failed");
        return -1;
    }
    mutex.shm_fd = -1;
    free(mutex.name);
    return 0;
}

void shared_mutex_lock(shared_mutex_t * mutex) {
    if (mutex->ptr == NULL) {
        perror("mutex is empty, lock file failed");
        return;
    }
    mutex->ptr->l_type = F_WRLCK;
    fcntl(mutex->shm_fd, F_SETLKW, mutex->ptr);
}

void shared_mutex_unlock(shared_mutex_t * mutex) {
    if (mutex->ptr == NULL) {
        perror("mutex is empty, unlock file failed");
        return;
    }
    mutex->ptr->l_type = F_UNLCK;
    fcntl(mutex->shm_fd, F_SETLKW, mutex->ptr);
}

static bool file_exists(const char * fname) {
    int fd = open(fname, O_RDWR | O_EXCL, 0666);
    if (fd < 0) {
        return false; 
    }
    close(fd);
    return true;
}

static void create_file(const char * fname) {
    int fd = open(fname, O_RDWR | O_CREAT, 0666);
    close(fd);
}

#elif defined(__linux__)  || defined(LINUX) // Linux

shared_mutex_t shared_mutex_init(const char* iname) {
    shared_mutex_t mutex = {NULL, 0, NULL, 1};
    std::string iname_md5 = md5(std::string(iname));
    char name[TNN_NAME_MAX];

    sprintf(name, ".%u.%s.tnnmutex", getuid(), iname_md5.c_str());

    // @brief open existing shared memory object, or create one.
    mutex.shm_fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0660);
    if (mutex.shm_fd < 0) {
        // sleep to wait ftruncated 
        usleep(100*1000);
        mutex.shm_fd = shm_open(name, O_RDWR | O_EXCL, 0660);
        mutex.created = 0;
    }

    if (mutex.created ) {
        if (ftruncate(mutex.shm_fd, sizeof(pthread_mutex_t)) != 0) {
            perror("rapidnet cache file ftruncate failed");
            return mutex;
        }
    }

    // Map pthread mutex into the shared memory.
    void *addr = mmap(NULL, sizeof(pthread_mutex_t), PROT_READ|PROT_WRITE,
        MAP_SHARED, mutex.shm_fd, 0);

    if (addr == MAP_FAILED) {
        perror("rapidnet cache file mmap failed");
        return mutex;
    }

    pthread_mutex_t *mutex_ptr = (pthread_mutex_t*)addr;

    // initialize the mutex.
    if (mutex.created) {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutexattr_setrobust(&attr,  PTHREAD_MUTEX_ROBUST);
        pthread_mutex_init(mutex_ptr, &attr);
    }

    mutex.ptr = mutex_ptr;
    mutex.name = (char*)malloc(TNN_NAME_MAX + 1);
    strcpy(mutex.name, name);

    return mutex;
}

int shared_mutex_close(shared_mutex_t mutex) {
    if (mutex.ptr != NULL) {
        if (munmap((void *)mutex.ptr, sizeof(pthread_mutex_t))) {
            perror("rapidnet cache file munmap failed");
            return -1;
        }
        mutex.ptr = NULL;
    }
    if (close(mutex.shm_fd)) {
        perror("rapidnet cache file close failed");
        return -1;
    }
    mutex.shm_fd = 0;
    free(mutex.name);
    return 0;
}

void shared_mutex_lock(shared_mutex_t * mutex) {
    int ret = pthread_mutex_lock(mutex->ptr);
    // make mutex consistent when last owner was crashed.
    if (ret == EOWNERDEAD) {
        pthread_mutex_consistent(mutex->ptr);
    }
}

void shared_mutex_unlock(shared_mutex_t * mutex) {
    pthread_mutex_unlock(mutex->ptr);
}

static bool file_exists(const char * fname) {
    int fd = open(fname, O_RDWR | O_EXCL, 0666);
    if (fd < 0) {
        return false; 
    }
    close(fd);
    return true;
}

static void create_file(const char * fname) {
    int fd = open(fname, O_RDWR | O_CREAT, 0666);
    close(fd);
}

#else

shared_mutex_t shared_mutex_init(const char* iname) {
    shared_mutex_t mutex = {NULL, 0, NULL, 1};
    // not support set robust
    // ExclFile need to resolve deadlock problem before used
    throw std::runtime_error("Shared mutex is not supported on current system");

    return mutex;
}

int shared_mutex_close(shared_mutex_t mutex) {
    return 0;
}

void shared_mutex_lock(shared_mutex_t * mutex) {
    // not support make mutex consistent
    // ExclFile need to fix when last owner was crashed before used
}

void shared_mutex_unlock(shared_mutex_t * mutex) {
}

static bool file_exists(const char * fname) {
    int fd = open(fname, O_RDWR | O_EXCL, 0666);
    if (fd < 0) {
        return false;
    }
    close(fd);
    return true;
}

static void create_file(const char * fname) {
    int fd = open(fname, O_RDWR | O_CREAT, 0666);
    close(fd);
}
#endif

ExclFile::ExclFile(std::string fname) : m_fname(fname) {
    static_mutex.lock();
    this->m_lock_name = this->m_fname + "~";
    this->m_created = false;
    this->m_mutex = shared_mutex_init(this->m_lock_name.c_str());
    // get mutex 
    shared_mutex_lock(&this->m_mutex);
}

ExclFile::~ExclFile() {
    if (this->m_created) {
        create_file(this->m_lock_name.c_str());
    }
    // release mutex
    shared_mutex_unlock(&this->m_mutex);
    shared_mutex_close(this->m_mutex);
    static_mutex.unlock();
}


bool ExclFile::Ready() {
    bool success;
    if (this->IsLockFileExists() && this->IsFileExists()) {
        success = true;
    }  else {
        success = false; 
        this->m_created = true;
    }
    return success;
}

bool ExclFile::IsFileExists() {
    return file_exists(this->m_fname.c_str());
}

bool ExclFile::IsLockFileExists() {
    return file_exists(this->m_lock_name.c_str());
}

}  //  namespace TNN_NS
