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

$PYTHON script/detect_dependency.py 

mkdir -p build
cd build && $CMAKE .. \
    -DCMAKE_CXX_COMPILER=$CPP_COMPILER \
    -DCMAKE_C_COMPILER=$C_COMPILER \
    -DPYTHON_EXECUTABLE=$PYTHON \
    && make -j4 &&  cd ..

cp build/*.so .

rm -rf build/
