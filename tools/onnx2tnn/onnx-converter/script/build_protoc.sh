cd $1
echo "in `pwd`"
# ~/cmake-3.6.3-Linux-x86_64/bin/cmake ../cmake/ -DBUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF -DCMAKE_CXX_COMPILER=g++-4.8 -DCMAKE_C_COMPILER=gcc-4.8
$CMAKE ../cmake/ \
 -DCMAKE_CXX_FLAGS=" -fPIC -fvisibility=hidden "\
 -DBUILD_SHARED_LIBS=OFF \
 -Dprotobuf_BUILD_TESTS=OFF \
 -Dprotobuf_BUILD_EXAMPLES=OFF \
 -DCMAKE_INSTALL_PREFIX=../../protobuf \
 -DCMAKE_CXX_COMPILER=$CPP_COMPILER \
 -DCMAKE_C_COMPILER=$C_COMPILER && make -j4 install && cp protoc ../../
