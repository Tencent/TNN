if [ ! -d "bin" ]; then
  mkdir bin
fi

cd bin

cmake ../../.. \
	-DCMAKE_BUILD_TYPE=Release \
	-DTNN_CPU_ENABLE:BOOL="ON" \
	-DTNN_MODEL_CHECK_ENABLE:BOOL="ON" 

make -j4

if [ -f "model_check" ]; then
	rm -rf CMake*
	rm -rf cmake*
	rm -rf Make*
	rm -rf source
	rm -rf tools
	
	cd ../onnx2tnn/onnx-converter
	./build.sh

	echo "Compiled successfully !"
else
	cd ..
	rm -rf bin
	echo "Compiled failed !!!"
fi

