TNN_VERSION_PATH=$PWD/version

echo ' '
echo '******************** step 1: update version.h ********************'
cd $TNN_VERSION_PATH
source $TNN_VERSION_PATH/version.sh
source $TNN_VERSION_PATH/add_version_attr.sh
cd $TNN_BUILD_PATH

echo ' '
echo '******************** step 2: start build rpn ********************'
#新建build目录
mkdir build
cd build
cmake ./../../
make -j 4

echo ' '
echo '******************** step 3: add version attr ********************'
#添加版本信息到库文件
# AddAllVersionAttr lib_full_path
