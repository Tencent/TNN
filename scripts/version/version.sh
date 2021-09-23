# add version to file
# 1:target 2:branch 3:date 4:hash  5：file
function AddToVersionFileH()
{
  echo "" >> $5
  echo "Target: "${1}
  echo "Commit Branch: "${2}
  echo "static char *branch_name_${1} = \""${2}"\";" >> $5
  echo "Commit Date: "${3}
  echo "static char *commit_date_${1} = \""${3}"\";" >> $5
  echo "Commit Hash: "${4}
  echo "static char *commit_hash_${1} = \""${4}"\";" >> $5
}

# add all git attr
# 1：lib file full path
function AddAllVersionAttr()
{
  if [[ ! -f "${1}" ]]; then
    echo "文件不存在: "${1}
    return 1
  fi

  AddVersionAttr $(echo ${VERSION_INFO_TNN[*]}) ${1}
}

# add all git attr
# 1：lib file full path
function AddAllVersion2Plist()
{
  if [[ ! -f "${1}" ]]; then
    echo "文件不存在: "${1}
    return 1
  fi

  PLIST_PATH=${1}
  SDK_INFO_KEY="YTSDKInfo"

  #修改plist的YTSDKInfo字段
  /usr/libexec/PlistBuddy -c "Delete  $SDK_INFO_KEY" $PLIST_PATH
  /usr/libexec/PlistBuddy -c "Add  $SDK_INFO_KEY dict" $PLIST_PATH

  AddVersion2Plist ${SDK_INFO_KEY} $(echo ${VERSION_INFO_TNN[*]}) ${1}
}

echo $PWD

TNN_VERSION_BUILD_PATH=$PWD
TNN_VERSION_FILE_PATH=$PWD/version.h

# 获取各个git信息
source $TNN_VERSION_BUILD_PATH/get_git_version.sh
VERSION_INFO_TNN=($(GetGitVersion $TNN_VERSION_BUILD_PATH tnn))

# 写入version.h文件
rm $TNN_VERSION_FILE_PATH -f
echo "// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the \"License\"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the 
// specific language governing permissions and limitations under the License." >> $TNN_VERSION_FILE_PATH
echo "#ifndef TNN_INCLUDE_TNN_VERSION_H_" >> $TNN_VERSION_FILE_PATH
echo "#define TNN_INCLUDE_TNN_VERSION_H_" >> $TNN_VERSION_FILE_PATH


AddToVersionFileH $(echo ${VERSION_INFO_TNN[*]}) $TNN_VERSION_FILE_PATH

echo "" >> $TNN_VERSION_FILE_PATH
echo "#endif //TNN_INCLUDE_TNN_VERSION_H_" >> version.h

cp version.h $TNN_VERSION_BUILD_PATH/../../include/tnn/

# 删除临时文件
rm version.h -f
