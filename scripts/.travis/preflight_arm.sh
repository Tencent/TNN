#!/bin/bash

set -e

git fetch origin master:master
CHANGED_FILES=`git diff --name-only master...${TRAVIS_COMMIT}`
ARM_RELEVANT=False

PATTERNS=("CMakeLists.txt"
          "cmake/"
          "include/tnn/"
          "scripts/build_android.sh"
          "source/tnn/core/"
          "source/tnn/device/arm/"
          "source/tnn/device/cpu/"
          "source/tnn/interpreter/"
          "source/tnn/layer/"
          "source/tnn/memory_manager/"
          "source/tnn/network/"
          "source/tnn/optimizer/"
          "source/tnn/utils/"
          )

echo "Patterns:"
echo ${PATTERNS[@]}

for CHANGED_FILE in $CHANGED_FILES; do
  for PATTERN in ${PATTERNS[@]}; do
    if [[ $CHANGED_FILE =~ $PATTERN ]]; then
      echo $CHANGED_FILE " -> MATCHES <- " $PATTERN
      ARM_RELEVANT=True
      break
    fi
  done
done

if [[ $ARM_RELEVANT == True ]]; then
  echo "Code changes relevant to arm, continuing with build."
else
  echo "Code changes not relevant to arm, exiting."
  exit 1
fi
