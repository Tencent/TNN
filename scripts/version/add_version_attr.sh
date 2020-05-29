# add git attr
# 1-4:version from func GetGitVersion 5：lib file full path
#
function AddVersionAttr()
{
  if [[ ! -f "${5}" ]]; then
    echo "文件不存在: "${5}
    return 1
  fi

  TARGET=${1}
  BRANCH=${2}
  DATE=${3}
  HASH=${4}
  BRANCH_KEY=${TARGET}_commit_branch
  DATE_KEY=${TARGET}_commit_date
  HASH_KEY=${TARGET}_commit_hash
  if [[ "$(uname)" == "Darwin" ]]; then
    xattr -w ${BRANCH_KEY} ${BRANCH} ${5}
    xattr -w ${DATE_KEY} ${DATE} ${5}
    xattr -w ${HASH_KEY} ${HASH} ${5}
  elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
    attr -s ${BRANCH_KEY} -V ${BRANCH} ${5} > /dev/null
    attr -s ${DATE_KEY} -V ${DATE} ${5} > /dev/null
    attr -s ${HASH_KEY} -V ${HASH} ${5} > /dev/null
  fi
}
