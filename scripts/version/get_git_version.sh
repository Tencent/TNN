# get git version
# 1:git目录  2：git标识
function GetGitVersion()
{
  if [[ ! -d "${1}" ]]; then
    echo "文件夹不存在: "${1}
  fi

  ORIG_PATH=$PWD
  cd $1

  GIT_BRANCH_NAME=$(eval "git symbolic-ref --short -q HEAD")
  GIT_COMMIT_DATE=$(eval "git log -1 --pretty=format:'%ad' --date=short")
  GIT_VERSION_DATE=$(eval "git log -1 --pretty=format:'%ad' --date=short")
  GIT_COMMIT_HASH=$(eval "git log -1 --pretty=format:'%h'")
  if [ "$GIT_BRANCH_NAME" = "" ]; then
    GIT_BRANCH_NAME='HEAD'
  fi

  # 输出
  VERSION_INFO=(${2} ${GIT_BRANCH_NAME} ${GIT_COMMIT_DATE} ${GIT_COMMIT_HASH})
  echo ${VERSION_INFO[*]}

  cd $ORIG_PATH
}
