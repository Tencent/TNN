models benchmark:
push all benchmark models to android device dir /data/local/tmp/benchmark-model, then run benchmark_models.sh, you will get all model benchmark cost time info.

layer benchmark:
run benchmark_layer.sh -h, you can get help info. below is some import info:
run benchmark_layer.sh --gtest_list_tests, you can get all layer benchmark list with parameters info, use --gtest_filter to filter layer benchmark. for example, run benchmark_layer.sh --gtest_filter=LayerTest/AddLayer*, you can benchmark add layer only;run benchmark_layer.sh --gtest_filter=LayerTest/AddLayerTest.AddLayer/0, you can benchmark add layer with one special parameter only.

