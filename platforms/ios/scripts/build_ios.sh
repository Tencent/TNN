#!/bin/sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
TNN_BUILD_PATH=$PWD/../project/ios
#设置文件
PLIST_PATH=$TNN_BUILD_PATH/tnn/Info.plist

SDK_VERSION=0.2.0
TARGET_NAME="tnn"
CONFIGURATION="Release"

TNN_VERSION_PATH=$PWD

echo ' '
echo '******************** update version.h ********************'
cd $TNN_VERSION_PATH
sh version.sh
cd $TNN_BUILD_PATH

echo ' '
echo '******************** start build rpn ********************'
#删除旧SDK文件
#rm -r ./${TARGET_NAME}.bundle
rm -r ./${TARGET_NAME}.framework
rm -r build

#更新Plist文件
SDK_INFO_KEY="YTSDKInfo"
GIT_BRANCH_NAME=$(eval "git symbolic-ref --short -q HEAD")
GIT_COMMIT_DATE=$(eval "git log -1 --pretty=format:'%ad' --date=format:'%Y-%m-%d %H:%M:%S'")
GIT_COMMIT_HASH=$(eval "git log -1 --pretty=format:'%h'")

#修改plist的YTSDKInfo字段
/usr/libexec/PlistBuddy -c "Delete  $SDK_INFO_KEY" $PLIST_PATH
/usr/libexec/PlistBuddy -c "Add  $SDK_INFO_KEY dict" $PLIST_PATH
/usr/libexec/PlistBuddy -c "Add  $SDK_INFO_KEY:hash string $GIT_COMMIT_HASH" $PLIST_PATH
/usr/libexec/PlistBuddy -c "Add  $SDK_INFO_KEY:date string $GIT_COMMIT_DATE" $PLIST_PATH

#更新版本号
agvtool new-marketing-version ${SDK_VERSION}
#更新build号
agvtool next-version -all


#编译 SDK
xcodebuild -target "$TARGET_NAME" -configuration ${CONFIGURATION}  -sdk iphoneos build
cp -r build/$CONFIGURATION-iphoneos/$TARGET_NAME.framework build
xcodebuild -target "$TARGET_NAME" -configuration ${CONFIGURATION}  -sdk iphonesimulator build
lipo -create "build/$CONFIGURATION-iphonesimulator/$TARGET_NAME.framework/$TARGET_NAME" "build/$CONFIGURATION-iphoneos/$TARGET_NAME.framework/$TARGET_NAME" -output "build/$TARGET_NAME.framework/$TARGET_NAME"
cp -r build/$TARGET_NAME.framework .
rm -r build

# 对于包含Metal的SDK, 转移metallib文件到bundle
if [ ! -d $TARGET_NAME.bundle ]; then
 mkdir $TARGET_NAME.bundle
fi

if [ ! -d $TARGET_NAME.framework/default.metallib ]; then
 cp $TARGET_NAME.framework/default.metallib $TARGET_NAME.bundle/$TARGET_NAME.metallib
 rm $TARGET_NAME.framework/default.metallib
fi
