# add version
# 1:git目录  2：git标识 3：version文件目录
function AddVersion()
{
  if [[ ! -d "${1}" ]]; then
    echo "文件夹不存在: "${1}
  	return
  fi

  ORIG_PATH=$PWD
  cd $1

  GIT_BRANCH_NAME=$(eval "git symbolic-ref --short -q HEAD")
  GIT_COMMIT_DATE=$(eval "git log -1 --pretty=format:'%ad' --date=format:'%Y-%m-%d %H:%M:%S'")
  GIT_VERSION_DATE=$(eval "git log -1 --pretty=format:'%ad' --date=format:'%Y%m%d%H%M'")
  GIT_COMMIT_HASH=$(eval "git log -1 --pretty=format:'%h'")

  echo "" >> $3

  echo "Target: "${2}
  echo "Commit Branch: "${GIT_BRANCH_NAME}
  echo "static char *branch_name_${2} = \""${GIT_BRANCH_NAME}"\";" >> $3
  echo "Commit Date: "${GIT_COMMIT_DATE}
  echo "static char *commit_date_${2} = \""${GIT_COMMIT_DATE}"\";" >> $3
  echo "Commit Hash: "${GIT_COMMIT_HASH}
  echo "static char *commit_hash_${2} = \""${GIT_COMMIT_HASH}"\";" >> $3

  cd $ORIG_PATH
}



TNN_VERSION_BUILD_PATH=$PWD
TNN_VERSION_FILE_PATH=$PWD/version.h
ARM_VERSION_BUILD_PATH=$PWD/../source/device/arm
CPU_VERSION_BUILD_PATH=$PWD/../source/device/cpu
CUDA_VERSION_BUILD_PATH=$PWD/../source/device/cuda
OPENCL_VERSION_BUILD_PATH=$PWD/../source/device/opencl
METAL_VERSION_BUILD_PATH=$PWD/../source/device/metal
X86_VERSION_BUILD_PATH=$PWD/../source/device/x86

rm $TNN_VERSION_FILE_PATH

echo "// Tencent is pleased to support the open source community by making TNN available.
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
// specific language governing permissions and limitations under the License." >> $TNN_VERSION_FILE_PATH
echo "#ifndef TNN_VERSION_H" >> $TNN_VERSION_FILE_PATH
echo "#define TNN_VERSION_H" >> $TNN_VERSION_FILE_PATH

AddVersion $TNN_VERSION_BUILD_PATH tnn $TNN_VERSION_FILE_PATH
AddVersion $ARM_VERSION_BUILD_PATH arm $TNN_VERSION_FILE_PATH
AddVersion $CPU_VERSION_BUILD_PATH cpu $TNN_VERSION_FILE_PATH
AddVersion $CUDA_VERSION_BUILD_PATH cuda $TNN_VERSION_FILE_PATH
AddVersion $OPENCL_VERSION_BUILD_PATH opencl $TNN_VERSION_FILE_PATH
AddVersion $METAL_VERSION_BUILD_PATH metal $TNN_VERSION_FILE_PATH
AddVersion $X86_VERSION_BUILD_PATH x86 $TNN_VERSION_FILE_PATH

echo "" >> $TNN_VERSION_FILE_PATH
echo "#endif //TNN_VERSION_H" >> version.h

cp version.h $TNN_VERSION_BUILD_PATH/../include

rm version.h
