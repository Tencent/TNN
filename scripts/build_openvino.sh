# TNN/scripts/
TNN_DIR=$(pwd)/../
thirdparty_dir=${TNN_DIR}/source/tnn/network/openvino/thirdparty/

if [ ! -d ${thirdparty_dir} ]
then
    mkdir ${thirdparty_dir}
    mkdir ${thirdparty_dir}/openvino
    mkdir ${thirdparty_dir}/openvino/lib
    mkdir ${thirdparty_dir}/ngraph
fi

# 编译 openvino 库
if [ ! -d ${TNN_DIR}/scripts/build_openvino ]
then
mkdir build_openvino
fi

cd build_openvino

if [ ! -d ${TNN_DIR}/scripts/build_openvino/openvinoInstall ]
then
# TNN/scripts/build_openvino
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git reset --hard 9df6a8f

# TNN/scripts/build_openvino/openvino
git submodule update --init --recursive

# 编译静态库
sed -i '152,152s/SHARED/STATIC/g' inference-engine/src/inference_engine/CMakeLists.txt
sed -i 's/SHARED/STATIC/g' inference-engine/src/legacy_api/CMakeLists.txt
sed -i 's/SHARED/STATIC/g' inference-engine/src/transformations/CMakeLists.txt
sed -i 's/SHARED/STATIC/g' inference-engine/src/low_precision_transformations/CMakeLists.txt
sed -i 's/SHARED/STATIC/g' ngraph/src/ngraph/CMakeLists.txt

mkdir build && cd build

# TNN/scripts/build_openvino/openvino/build
cmake ../ \
-DENABLE_OPENCV=OFF \
-DCMAKE_INSTALL_PREFIX=${TNN_DIR}/scripts/build_openvino/openvinoInstall \
-DENABLE_CLDNN=OFF \
-DENABLE_TBB_RELEASE_ONLY=OFF \
-DTHREADING=SEQ \
-DNGRAPH_COMPONENT_PREFIX="deployment_tools/ngraph/" \
-DENABLE_MYRIAD=OFF \
-DNGRAPH_JSON_ENABLE=OFF \
-DENABLE_PROFILING_ITT=OFF \

make -j4
make install
cd ../../

# TNN/scripts/build_openvino/ 拷贝 lib 和 include 文件到 thirdparty 下
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
cmake ../../ \
-DTNN_OPENVINO_ENABLE=ON \
-DTNN_X86_ENABLE=ON \
-DTNN_TEST_ENABLE=ON \
-DTNN_CPU_ENABLE=ON \

make -j4