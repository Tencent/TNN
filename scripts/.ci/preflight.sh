#!/bin/bash

set -e

ci_type=$1

git fetch origin master:master
CHANGED_FILES=`git diff --name-only master`
echo -e "\n>> Changed Files:"
for CHANGED_FILE in $CHANGED_FILES; do
  echo ${CHANGED_FILE}
done
RELEVANT=False

PATTERNS=("CMakeLists.txt"
          "cmake/"
          "include/tnn/"
          "source/tnn/core/"
          "source/tnn/device/cpu/"
          "source/tnn/interpreter/"
          "source/tnn/layer/"
          "source/tnn/memory_manager/"
          "source/tnn/network/"
          "source/tnn/optimizer/"
          "source/tnn/utils/"
          "test/"
          )

if [[ ${ci_type} == 'android' ]]; then
  PATTERNS+=("platforms/android/"
             "scripts/build_android.sh"
             "source/tnn/device/arm/"
             "source/tnn/device/opencl/"
             )
elif [[ ${ci_type} == 'arm' ]]; then
  PATTERNS+=("scripts/build_aarch64_linux.sh"
             "scripts/build_armhf_linux.sh"
             "scripts/build_test.sh"
             "source/tnn/device/arm/"
             )
elif [[ ${ci_type} == 'ios' ]]; then
  PATTERNS+=("platforms/ios/"
             "scripts/build_framework_ios.sh"
             "source/tnn/device/arm/"
             "source/tnn/device/metal/"
             )
elif [[ ${ci_type} == 'x86' ]]; then
  PATTERNS+=("scripts/build_linux.sh"
             "scripts/build_macos.sh"
             "source/tnn/device/x86/"
             "source/tnn/extern_wrapper/"
             )
else
  PATTERNS+=("*")
fi

echo -e "\n>> Patterns:"
for PATTERN in ${PATTERNS[@]}; do
  echo ${PATTERN}
done
echo ""

for CHANGED_FILE in $CHANGED_FILES; do
  for PATTERN in ${PATTERNS[@]}; do
    if [[ $CHANGED_FILE =~ $PATTERN ]]; then
      echo $CHANGED_FILE " -> MATCHES <- " $PATTERN
      RELEVANT=True
      break
    fi
  done
  if [[ $RELEVANT == True ]]; then
    break
  fi
done

if [[ $RELEVANT == True ]]; then
  echo "Code changes relevant to" ${ci_type} ", continuing with build."
else
  echo "Code changes not relevant to" ${ci_type} ", exiting."
  exit 11
fi
