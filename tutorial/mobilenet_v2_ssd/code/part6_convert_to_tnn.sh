cd ../../../tools/convert2tnn/

if [ ! -d "bin" ]; then
  echo "TNN model conversion tool is not compiled!"
  exit
fi

python3 converter.py onnx2tnn ../../tutorial/mobilenet_v2_ssd/code/new_model.onnx -align