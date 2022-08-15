python3 -m tf2onnx.convert  \
  --graphdef saved_model.pb \
  --inputs "Preprocessor/sub:0[1,300,300,3]" \
  --inputs-as-nchw "Preprocessor/sub:0" \
  --outputs "Postprocessor/convert_scores:0,concat:0" \
  --output saved_model.onnx \
  --opset 11 \
  --fold_const