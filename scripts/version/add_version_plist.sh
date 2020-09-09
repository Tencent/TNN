# add version to plist
# 1:parent key
# 2-5:version from func GetGitVersion
# 6：lib file full path
function AddVersion2Plist()
{
  if [[ ! -f "${6}" ]]; then
    echo "文件不存在: "${6}
    return 1
  fi

  SDK_INFO_KEY=${1}

  TARGET=${2}
  BRANCH=${3}
  DATE=${4}
  HASH=${5}

  BRANCH_KEY=${TARGET}_commit_branch
  DATE_KEY=${TARGET}_commit_date
  HASH_KEY=${TARGET}_commit_hash

  /usr/libexec/PlistBuddy -c "Add  $SDK_INFO_KEY:${BRANCH_KEY} string ${BRANCH}" ${6}
  /usr/libexec/PlistBuddy -c "Add  $SDK_INFO_KEY:${DATE_KEY} string ${DATE}" ${6}
  /usr/libexec/PlistBuddy -c "Add  $SDK_INFO_KEY:${HASH_KEY} string ${HASH}" ${6}
}
