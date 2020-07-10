# TNN/scripts/
TNN_DIR=$(pwd)/../
thirdparty_dir=${TNN_DIR}/source/tnn/device/openvino/thirdparty/
mkdir ${thirdparty_dir}
mkdir ${thirdparty_dir}/openvino
mkdir ${thirdparty_dir}/openvino/lib
mkdir ${thirdparty_dir}/ngraph

mkdir openvinoWrapper && cd openvinoWrapper

# TNN/scripts/openvinoWrapper
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino

# TNN/scripts/openvinoWrapper/openvino
git submodule update --init --recursive
mkdir build && cd build

# TNN/scripts/openvinoWrapper/openvino/build
cmake ../ \
-DENABLE_OPENCV=OFF \
-DCMAKE_INSTALL_PREFIX=${TNN_DIR}/scripts/openvinoWrapper/openvinoInstall \
-DENABLE_CLDNN=OFF \
-DENABLE_TBB_RELEASE_ONLY=OFF \
-DTHERDING=SEQ \
-DNGRAPH_COMPONENT_PREFIX="deployment_tools/ngraph/" \

make -j8
make install
cd ../../

# TNN/scripts/openvinoWrapper/
cp -r openvinoInstall/deployment_tools/inference_engine/include/ ${thirdparty_dir}/openvino/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/* ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/ngraph/include/ ${thirdparty_dir}/ngraph/
cp -r openvinoInstall/deployment_tools/ngraph/lib64/* ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/external/tbb/lib/* ${thirdparty_dir}/openvino/lib/
cp openvinoInstall/lib64/libpugixml.a ${thirdparty_dir}/openvino/lib/

cmake ../../ \
-DTNN_OPENVINO_ENABLE=ON \
-DTNN_TEST_ENABLE=ON \
-DTNN_CPU_ENABLE=ON \
-DDEBUG=ON \

make -j8