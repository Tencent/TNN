# TNN/scripts/
# 判断 cmake version
if !(command -v cmake > /dev/null 2>&1); then
    echo "Cmake not found!"
    exit 1
fi

for var in $(cmake --version | awk 'NR==1{print $3}')
do
    cmake_version=$var
done
function version_lt { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"; }

if (version_lt $cmake_version 3.11); then
    echo "Cmake 3.11 or higher is required. You are running version ${cmake_version}"
    exit 2
fi

TNN_DIR=$(pwd)/../
thirdparty_dir=${TNN_DIR}/source/tnn/network/openvino/thirdparty/
export GIT_LFS_SKIP_SMUDGE=1

if [ ! -d ${thirdparty_dir} ]
then
    mkdir ${thirdparty_dir}
    mkdir ${thirdparty_dir}/openvino
    mkdir ${thirdparty_dir}/openvino/lib
    mkdir ${thirdparty_dir}/ngraph
fi

# 编译 openvino 库
if [ ! -d ${TNN_DIR}/scripts/build_linux ]
then
mkdir build_linux
fi

cd build_linux

if [ ! -d ${TNN_DIR}/scripts/build_linux/openvinoInstall ]
then
# TNN/scripts/build_linux
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git reset --hard 9df6a8f

# TNN/scripts/build_linux/openvino
git submodule update --init --recursive

# 编译静态库
sed -i '152,152s/SHARED/STATIC/g' inference-engine/src/inference_engine/CMakeLists.txt
sed -i 's/SHARED/STATIC/g' inference-engine/src/legacy_api/CMakeLists.txt
sed -i 's/SHARED/STATIC/g' inference-engine/src/transformations/CMakeLists.txt
sed -i 's/SHARED/STATIC/g' inference-engine/src/low_precision_transformations/CMakeLists.txt
sed -i 's/SHARED/STATIC/g' ngraph/src/ngraph/CMakeLists.txt

mkdir build && cd build

echo "Configuring Openvino ..."
# TNN/scripts/build_linux/openvino/build
cmake ../ \
-DENABLE_OPENCV=OFF \
-DCMAKE_INSTALL_PREFIX=${TNN_DIR}/scripts/build_linux/openvinoInstall \
-DENABLE_CLDNN=OFF \
-DENABLE_TBB_RELEASE_ONLY=OFF \
-DTHREADING=SEQ \
-DNGRAPH_COMPONENT_PREFIX="deployment_tools/ngraph/" \
-DENABLE_MYRIAD=OFF \
-DNGRAPH_JSON_ENABLE=OFF \
-DENABLE_PROFILING_ITT=OFF \

echo "Building Openvino ..."
make -j4
make install
cd ../../

# TNN/scripts/build_linux/ 拷贝 lib 和 include 文件到 thirdparty 下
cp -r openvinoInstall/deployment_tools/inference_engine/include/ ${thirdparty_dir}/openvino/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/libinference_engine.a ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/libinference_engine_legacy.a ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/libinference_engine_transformations.a ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/libinference_engine_lp_transformations.a ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/plugins.xml ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/plugins.xml ./
cp -r openvinoInstall/deployment_tools/ngraph/include/ ${thirdparty_dir}/ngraph/
if [ -d openvinoInstall/deployment_tools/ngraph/lib64/ ]
then
cp -r openvinoInstall/deployment_tools/ngraph/lib64/libngraph.a ${thirdparty_dir}/openvino/lib/
else
cp -r openvinoInstall/deployment_tools/ngraph/lib/libngraph.a ${thirdparty_dir}/openvino/lib/
fi
if [ -d openvinoInstall/lib64/ ]
then
cp openvinoInstall/lib64/libpugixml.a ${thirdparty_dir}/openvino/lib/
else
cp openvinoInstall/lib/libpugixml.a ${thirdparty_dir}/openvino/lib/
fi
fi # openvinoInstall

# 编译 TNN
echo "Configuring TNN ..."
cmake ../../ \
-DTNN_OPENVINO_ENABLE=ON \
-DTNN_X86_ENABLE=ON \
-DTNN_TEST_ENABLE=ON \
-DTNN_CPU_ENABLE=ON \

echo "Building TNN ..."
make -j4

echo "Done"