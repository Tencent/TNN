#!/bin/sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

if [ -z $TNN_ROOT_PATH ]
then
  TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

TNN_BUILD_PATH=$TNN_ROOT_PATH/platforms/ios
#设置文件
PLIST_PATH=$TNN_BUILD_PATH/tnn/Info.plist
TNN_VERSION_PATH=$TNN_ROOT_PATH/scripts/version

# iPhone、iPhone+Simulator、 Mac
DEVICE_PLATFORM="iPhone+Simulator"
# DEVICE_PLATFORM="Mac"

SDK_VERSION=0.2.0
TARGET_NAME="tnn"
CONFIGURATION="Release"
XCODE_VERSION=`xcodebuild -version | awk 'NR == 1 {print $2}'`
XCODE_MAJOR_VERSION=`echo $XCODE_VERSION | awk -F. '{print $1}'`


echo ' '
echo '******************** step 1: update version.h ********************'
cd $TNN_VERSION_PATH
source $TNN_VERSION_PATH/version.sh
source $TNN_VERSION_PATH/add_version_attr.sh
cd $TNN_BUILD_PATH

echo ' '
echo '******************** step 2: start build rpn ********************'
#删除旧SDK文件
#rm -r ./${TARGET_NAME}.bundle
rm -r ./${TARGET_NAME}.framework
rm -r build


#更新版本号
agvtool new-marketing-version ${SDK_VERSION}
#更新build号
agvtool next-version -all
#更新Plist文件
source $TNN_VERSION_PATH/add_version_plist.sh
AddAllVersion2Plist $PLIST_PATH


#编译 SDK
if [[ $DEVICE_PLATFORM == iPhone* ]]; then
  echo ' '
  echo '******************** Build iPhone SDK ********************'
  # 指定 arm64
  # xcodebuild -target "$TARGET_NAME" -configuration ${CONFIGURATION}  -sdk iphoneos -arch arm64 build
  xcodebuild -quiet -target "$TARGET_NAME" -configuration ${CONFIGURATION}  -sdk iphoneos build CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO
  cp -r build/$CONFIGURATION-iphoneos/$TARGET_NAME.framework build
elif [ $DEVICE_PLATFORM == "Mac" ]; then
  echo ' '
  echo '******************** Build Mac SDK ********************'
  xcodebuild -quiet -target "$TARGET_NAME" -configuration ${CONFIGURATION}  -sdk macosx -arch x86_64 build CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO
  cp -r build/$CONFIGURATION/$TARGET_NAME.framework build
fi


if [ $DEVICE_PLATFORM == "iPhone+Simulator" ]; then
  echo ' '
  echo '******************** Build Simulator SDK ********************'
  # 指定 i386
  # xcodebuild -target "$TARGET_NAME" -configuration ${CONFIGURATION}  -sdk iphonesimulator -arch i386 build
  if [ $XCODE_MAJOR_VERSION -ge 12 ]; then
      xcodebuild -quiet -target "$TARGET_NAME" -configuration ${CONFIGURATION}  -sdk iphonesimulator EXCLUDED_ARCHS=arm64 build CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO
  else
      xcodebuild -quiet -target "$TARGET_NAME" -configuration ${CONFIGURATION}  -sdk iphonesimulator build CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO
  fi
  # merge lib
  lipo -create "build/$CONFIGURATION-iphonesimulator/$TARGET_NAME.framework/$TARGET_NAME" "build/$CONFIGURATION-iphoneos/$TARGET_NAME.framework/$TARGET_NAME" -output "build/$TARGET_NAME.framework/$TARGET_NAME"
  # copy metallib
  cp -r "build/$CONFIGURATION-iphonesimulator/$TARGET_NAME.framework/default.metallib" "build/$TARGET_NAME.framework/default.simulator.metallib"
fi

cp -r build/$TARGET_NAME.framework .
rm -r build

# 对于包含Metal的SDK, 转移metallib文件到bundle
if [ ! -d $TARGET_NAME.bundle ]; then
 mkdir $TARGET_NAME.bundle
fi

if [ ! -d $TARGET_NAME.framework/default.metallib ]; then
  cp $TARGET_NAME.framework/default.metallib $TARGET_NAME.bundle/${TARGET_NAME}.metallib
  rm $TARGET_NAME.framework/default.metallib
  cp $TARGET_NAME.framework/default.simulator.metallib $TARGET_NAME.bundle/${TARGET_NAME}.simulator.metallib
  rm $TARGET_NAME.framework/default.simulator.metallib
fi

echo ' '
echo '******************** step 3: add version attr ********************'
#添加版本信息到库文件
AddAllVersionAttr "$TNN_BUILD_PATH/$TARGET_NAME.framework/$TARGET_NAME"
AddAllVersionAttr "$TNN_BUILD_PATH/$TARGET_NAME.bundle/$TARGET_NAME.metallib"

if [ ! -f $TARGET_NAME.framework/${TARGET_NAME} ]; then
    echo 'Error: building failed.'
    exit -1
fi

if [ ! -f $TARGET_NAME.bundle/${TARGET_NAME}.metallib ]; then
    echo 'Error: building metallib failed.'
    exit -1
fi
echo 'building completes.'
