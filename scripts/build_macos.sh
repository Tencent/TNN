mkdir macosbuild
cd macosbuild
cmake .. -DTNN_METAL_ENABLE=ON -DTNN_TEST_ENABLE=ON -DTNN_UNIT_TEST_ENABLE=ON
make -j4

#echo "build finished!!!"
#find . -name tnn.metallib
#
#./test/unit_test/unit_test -dt METAL -lp tnn.metallib

