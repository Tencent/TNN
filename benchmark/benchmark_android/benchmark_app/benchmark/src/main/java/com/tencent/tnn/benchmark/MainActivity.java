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
        String load_list = bundle.getString(ARGS_INTENT_KEY_LOAD_LIST, "libtnn_wrapper.so");
        Log.e("benchmark", benchmark_dir);
        Log.e("benchmark", load_list);
        for(String element : load_list.split(";")) {
            FileUtils.copyFile(benchmark_dir + "/" + element, getFilesDir().getAbsolutePath() + "/" + element);
            System.load(getFilesDir().getAbsolutePath() + "/libtnn_wrapper.so");
        }
        String args = bundle.getString(ARGS_INTENT_KEY_ARGS_0, bundle.getString(ARGS_INTENT_KEY_ARGS_1));
        String fileDir = initModel();

        Log.i(TAG, "Running TNN Benchmark with args: " + args);
        benchmark.nativeRun(args, fileDir);
    }

    private String initModel() {

        String targetDir = this.getFilesDir().getAbsolutePath();
        String[] files;
        try {
            files = this.getResources().getAssets().list("");
        } catch (IOException exception){
            Log.e(TAG, "Get TNN Benchmark assets failed");
            return targetDir;
        }

        for (int i = 0; i < files.length; i++) {
            if (files[i].contains(".tnnproto") || files[i].contains(".tnnmodel")) {
                String interModelFilePath = targetDir + "/" + files[i];
                boolean ret = FileUtils.copyAsset(this.getResources().getAssets(), files[i], interModelFilePath);
            }
        }
        return targetDir;
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

}
