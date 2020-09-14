export CMAKE=/path/to/cmake
export CPP_COMPILER=/path/to/g++
export C_COMPILER=/path/to/gcc
export PYTHON=/path/to/python3

# here is an example, make sure libpython3 is in LD_LIBRARY_PATH
export CMAKE=cmake
export CPP_COMPILER=g++
export C_COMPILER=gcc
export PYTHON=`which python3`

set -xe

if [ -d "build" ]; then
	rm -rf build/
fi


mkdir -p build
cd build && $CMAKE ../../.. \
		-DCMAKE_BUILD_TYPE=Release \
		-DTNN_CONVERTER_ENABLE="ON" \
		-DTNN_CPU_ENABLE="ON" \
		-DTNN_BUILD_SHARED="OFF" \


make -j4
cp tools/converter/TnnConverter ../../convert2tnn/bin/
cd ..
rm -rf build/




