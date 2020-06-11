if [ ! -d "bin" ]; then
  mkdir bin
fi

if [ ! -d "temp" ]; then
  mkdir temp
fi

cd temp

cmake ../../.. \
	-DCMAKE_BUILD_TYPE=Release \
	-DTNN_CPU_ENABLE:BOOL="ON" \
	-DTNN_MODEL_CHECK_ENABLE:BOOL="ON" 

make -j4

if [ $? -eq 0 ]; then
	mv model_check ../bin/
	cd ..
	rm -rf temp
	
	cd ../onnx2tnn/onnx-converter
	./build.sh
	
	echo "Compiled successfully !"
else
	cd ..
	rm -rf temp bin
	echo "Compiled failed !!!"
fi

