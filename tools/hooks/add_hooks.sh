SRC_HOOKS_DIR=$(pwd)
DOT_GIT_DIR=${SRC_HOOKS_DIR}/../../.git
DOT_GIT_HOOKS_DIR=${DOT_GIT_DIR}/hooks

if [ ! -d ${DOT_GIT_HOOKS_DIR} ]; then
  mkdir ${DOT_GIT_HOOKS_DIR}
fi

cp ${SRC_HOOKS_DIR}/pre-commit ${DOT_GIT_HOOKS_DIR}/pre-commit
