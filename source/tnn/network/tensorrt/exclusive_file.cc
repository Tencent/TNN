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

#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "exclusive_file.h"
#include "md5.h"

namespace TNN_NS {

static pthread_mutex_t static_mutex = PTHREAD_MUTEX_INITIALIZER;

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

ExclFile::ExclFile(std::string fname) : m_fname(fname) {
    pthread_mutex_lock(&static_mutex);
    this->m_lock_name = this->m_fname + "~";
    this->m_created = false;
    this->m_mutex = shared_mutex_init(this->m_lock_name.c_str());
    // get mutex 
    int ret = pthread_mutex_lock(this->m_mutex.ptr);
    // make mutex consistent when last owner was crashed.
    if (ret == EOWNERDEAD) {
        pthread_mutex_consistent(this->m_mutex.ptr);
    }
}

ExclFile::~ExclFile() {
    if (this->m_created) {
        int fd = open(this->m_lock_name.c_str(), O_RDWR | O_CREAT, 0666);
        close(fd);
    }
    // release mutex
    pthread_mutex_unlock(this->m_mutex.ptr);
    shared_mutex_close(this->m_mutex);
    pthread_mutex_unlock(&static_mutex);
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
    int fd = open(this->m_fname.c_str(), O_RDWR | O_EXCL, 0666);
    if (fd < 0) {
        return false; 
    }
    close(fd);
    return true;
}


bool ExclFile::IsLockFileExists() {
    int fd = open(this->m_lock_name.c_str(), O_RDWR | O_EXCL, 0666);
    if (fd < 0) {
        return false; 
    }
    close(fd);
    return true;
}

}  //  TNN_NS