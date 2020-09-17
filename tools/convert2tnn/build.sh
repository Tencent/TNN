if [ ! -d "bin" ]; then
  mkdir bin
fi

if [ ! -d "temp" ]; then
  mkdir temp
fi

function build_model_check() {
	cd $2

	cmake ../../.. \
		-DCMAKE_BUILD_TYPE=Release \
		-DTNN_CPU_ENABLE:BOOL="ON" \
		-DTNN_MODEL_CHECK_ENABLE:BOOL="ON" \
    -DTNN_CONVERTER_ENABLE:BOOL="ON" \
		-DTNN_BUILD_SHARED="OFF"

	make -j4
        cp tools/converter/TnnConverter ../bin/

	if [ -f "model_check" ]; then
		mv model_check ../$1

		cd ..

		rm -rf $2
		
		cd ../onnx2tnn/onnx-converter
		./build.sh

		echo "Compiled successfully !"
	else
		cd ..
		rm -rf $1 $2
		echo "Compiled failed !!!"
	fi
}

build_model_check bin temp

