# TNN/scripts/
TNN_DIR=$(pwd)/../
thirdparty_dir=${TNN_DIR}/source/tnn/device/openvino/thirdparty/

if [ ! -d ${thirdparty_dir} ]
then
    mkdir ${thirdparty_dir}
    mkdir ${thirdparty_dir}/openvino
    mkdir ${thirdparty_dir}/openvino/lib
    mkdir ${thirdparty_dir}/ngraph
fi

mkdir build_openvino && cd build_openvino

# TNN/scripts/build_openvino
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino

# TNN/scripts/build_openvino/openvino
git submodule update --init --recursive

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

make -j4
make install
cd ../../

# TNN/scripts/build_openvino/
cp -r openvinoInstall/deployment_tools/inference_engine/include/ ${thirdparty_dir}/openvino/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/libinference_engine.a ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/libinference_engine_legacy.a ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/libinference_engine_transformations.a ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/libinference_engine_lp_transformations.a ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/plugins.xml ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/inference_engine/lib/intel64/plugins.xml ./
cp -r openvinoInstall/deployment_tools/ngraph/include/ ${thirdparty_dir}/ngraph/
cp -r openvinoInstall/deployment_tools/ngraph/lib64/libngraph.a ${thirdparty_dir}/openvino/lib/
cp -r openvinoInstall/deployment_tools/ngraph/lib/libngraph.a ${thirdparty_dir}/openvino/lib/
cp openvinoInstall/lib64/libpugixml.a ${thirdparty_dir}/openvino/lib/
cp openvinoInstall/lib/libpugixml.a ${thirdparty_dir}/openvino/lib/

rm -rf openvinoInstall/deployment_tools/inference_engine/samples/
rm -rf openvino

cmake ../../ \
-DTNN_OPENVINO_ENABLE=ON \
-DTNN_TEST_ENABLE=ON \
-DTNN_CPU_ENABLE=ON \
-DDEBUG=ON \

make -j4