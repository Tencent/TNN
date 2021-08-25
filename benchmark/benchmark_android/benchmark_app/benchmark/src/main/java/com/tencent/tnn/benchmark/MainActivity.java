package com.tencent.tnn.benchmark;


import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import java.io.File;

public class MainActivity extends Activity {

    private TextView lightLiveCheckBtn;

    private static final String TAG = "TNN_BenchmarkModelActivity";
    private BenchmarkModel benchmark = new BenchmarkModel();
    private static final String ARGS_INTENT_KEY_ARGS_0 = "args";
    private static final String ARGS_INTENT_KEY_ARGS_1 = "--args";
    private static final String ARGS_INTENT_KEY_BENCHMARK_DIR = "benchmark-dir";
    private static final String ARGS_INTENT_KEY_LOAD_LIST = "load-list";
    private static final String ARGS_INTENT_KEY_MODEL = "model";

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        init();

    }

    private void init() {
        String model = "";
        try {
            Intent intent = getIntent();
            Bundle bundle = intent.getExtras();
            String benchmark_dir = bundle.getString(ARGS_INTENT_KEY_BENCHMARK_DIR, "/data/local/tmp/tnn-benchmark/");
            String[] load_list = bundle.getStringArray(ARGS_INTENT_KEY_LOAD_LIST);
            model = bundle.getString(ARGS_INTENT_KEY_MODEL);
            for(String element : load_list) {
                FileUtils.copyFile(benchmark_dir + "/" + element, getFilesDir().getAbsolutePath() + "/" + element);
                System.load(getFilesDir().getAbsolutePath() + "/" + element);
            }
            final String args = bundle.getString(ARGS_INTENT_KEY_ARGS_0, bundle.getString(ARGS_INTENT_KEY_ARGS_1));
            final String file_dir  = this.getFilesDir().getAbsolutePath();

            String tnnproto = model;
            String src_tnnproto_path = benchmark_dir + "/" + "benchmark-model/" + tnnproto;
            String target_tnnproto_output_path = file_dir + "/" + tnnproto;
            File tnnproto_file = new File(target_tnnproto_output_path);
            if (tnnproto_file.exists()) {
                tnnproto_file.delete();
            }
            tnnproto_file.createNewFile();
            FileUtils.copyFile(src_tnnproto_path, target_tnnproto_output_path);

            String tnnmodel = model.substring(0, model.length() - 5) + "model";
            String src_tnnmodel_path = benchmark_dir + "/" + "benchmark-model/" + tnnmodel;
            do {
                File test_file = new File(src_tnnmodel_path);
                if (!test_file.exists())
                    break;

                String target_tnnmodel_output_path = file_dir + "/" + tnnmodel;
                File tnnmodel_file = new File(target_tnnmodel_output_path);
                if (tnnmodel_file.exists()) {
                    tnnmodel_file.delete();
                }
                tnnmodel_file.createNewFile();
                FileUtils.copyFile(src_tnnmodel_path, target_tnnmodel_output_path);

            } while (false);

            int result = benchmark.nativeRun(args, file_dir);
            if(result != 0) {
                Log.i("tnn", String.format(" %s TNN Benchmark time cost failed error code: %d \n", model , result));
            }
        } catch(Error | Exception e) {
            Log.i("tnn", String.format(" %s TNN Benchmark time cost failed error/exception: %s \n", model, e.getMessage()));
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

}
