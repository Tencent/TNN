package com.tencent.tnn.benchmark;


import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Debug;
import android.view.View;
import android.widget.TextView;

import android.util.Log;
import com.tencent.tnn.benchmark.BenchmarkModel;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

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
        Intent intent = getIntent();
        Bundle bundle = intent.getExtras();
        String benchmark_dir = bundle.getString(ARGS_INTENT_KEY_BENCHMARK_DIR, "/data/local/tmp/tnn-benchmark/");
        String[] load_list = bundle.getStringArray(ARGS_INTENT_KEY_LOAD_LIST);
        for(String element : load_list) {
            FileUtils.copyFile(benchmark_dir + "/" + element, getFilesDir().getAbsolutePath() + "/" + element);
            System.load(getFilesDir().getAbsolutePath() + "/" + element);
        }
        String model = bundle.getString(ARGS_INTENT_KEY_MODEL);
        FileUtils.copyFile(benchmark_dir + "/" + "benchmark-model/" + model, getFilesDir().getAbsolutePath() + "/" + model);

        String args = bundle.getString(ARGS_INTENT_KEY_ARGS_0, bundle.getString(ARGS_INTENT_KEY_ARGS_1));

        Log.i(TAG, "Running TNN Benchmark with args: " + args);
        benchmark.nativeRun(args, this.getFilesDir().getAbsolutePath());
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

}
