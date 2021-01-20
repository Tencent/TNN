#!/bin/bash

set -e

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
          "scripts/build_linux.sh"
          "scripts/build_macos.sh"
          "source/tnn/core/"
          "source/tnn/device/cpu/"
          "source/tnn/device/x86/"
          "source/tnn/extern_wrapper/"
          "source/tnn/interpreter/"
          "source/tnn/layer/"
          "source/tnn/memory_manager/"
          "source/tnn/network/"
          "source/tnn/optimizer/"
          "source/tnn/utils/"
          )

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
done

if [[ $RELEVANT == True ]]; then
  echo "Code changes relevant, continuing with build."
else
  echo "Code changes not relevant, exiting."
  exit 1
fi
