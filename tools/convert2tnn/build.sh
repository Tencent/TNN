if [ ! -d "bin" ]; then
  mkdir bin
fi

function build_model_check() {
	cd $1

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
		
		cd ../../onnx2tnn/onnx-converter
		./build.sh

		echo "Compiled successfully !"
	else
		cd ..
		rm -rf $1
		echo "Compiled failed !!!"
	fi
}

build_model_check bin


