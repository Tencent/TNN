package com.tencent.tnn.demo.common.component;

import android.app.Dialog;
import android.content.Context;
import android.os.Bundle;
import android.text.TextPaint;
import android.view.View;
import android.view.Window;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.tencent.tnn.demo.R;


public class CustomDialog extends Dialog implements View.OnClickListener {
    private String mDialogTitle;
    private String mDialogTip;
    private String mButtonYes;
    private String mButtonNo;

    public CustomDialog(Context context) {
        super(context);
    }

    public CustomDialog(Context context, int themeResId) {
        super(context, themeResId);
    }

    public CustomDialog(Context context, boolean cancelable, OnCancelListener cancelListener) {
        super(context, cancelable, cancelListener);
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);

        setContentView(R.layout.dialog_layout);
        LinearLayout ll = (LinearLayout) findViewById(R.id.avd_root_view);

        FrameLayout.LayoutParams lp1 = (FrameLayout.LayoutParams) ll.getLayoutParams();
        lp1.width = getWindow().getWindowManager().getDefaultDisplay().getWidth() - 2 * dip2px(getContext(), 30);
        ll.setLayoutParams(lp1);

        TextView tvTitle = (TextView) findViewById(R.id.avd_dialog_title);
        tvTitle.setText(mDialogTitle);

        TextView tv1 = (TextView) findViewById(R.id.avd_dialog_tip);
        tv1.setText(mDialogTip);

        TextView mFirstButton = (TextView) findViewById(R.id.avd_button_yes);
        mFirstButton.setText(mButtonYes);

        TextPaint paint = mFirstButton.getPaint();
        float y1 = paint.measureText(mButtonYes);

        TextView mSecondButton = (TextView) findViewById(R.id.avd_button_no);
        mSecondButton.setText(mButtonNo);

        float y2;
        if (mButtonNo != null) {
            y2 = paint.measureText(mButtonNo);
        } else {
            y2 = 0;
        }

        float delta = dip2px(getContext(), 30) * 2;
        float y = Math.max(y1, y2) + delta;

        LinearLayout.LayoutParams lp = (LinearLayout.LayoutParams) mFirstButton.getLayoutParams();
        //lp.width = (int) y;
        mFirstButton.setLayoutParams(lp);

        LinearLayout.LayoutParams lp2 = (LinearLayout.LayoutParams) mSecondButton.getLayoutParams();
        //lp2.width = (int) y;
        mSecondButton.setLayoutParams(lp2);

        mSecondButton.setOnClickListener(this);
        mFirstButton.setOnClickListener(this);

        mFirstButton.setVisibility(View.GONE);
        mSecondButton.setVisibility(View.GONE);

        if (mButtonYes != null) {
            mFirstButton.setVisibility(View.VISIBLE);
        }

        if (mButtonNo != null) {
            mSecondButton.setVisibility(View.VISIBLE);
        }

        this.setCanceledOnTouchOutside(false);

    }

    public CustomDialog setTitle(String text) {
        mDialogTitle = text;
        return this;
    }

    public CustomDialog setTips(String text) {
        mDialogTip = text;
        return this;
    }

    public CustomDialog setPositiveText(String text) {
        mButtonYes = text;
        return this;
    }

    public CustomDialog setNegativeText(String text) {
        mButtonNo = text;
        return this;
    }

    public void setOnClickListener(DialogListener listener) {
        mListener = listener;
    }

    @Override
    public void onClick(View v) {

        int id = v.getId();
        if (id == R.id.avd_button_yes) {
            mListener.onYesClick();
        } else if (id == R.id.avd_button_no) {
            mListener.onNoClick();
        }
        dismiss();
    }


    private DialogListener mListener;

    public interface DialogListener {
        void onYesClick();

        void onNoClick();
    }

    private static int dip2px(Context context, float dpValue) {
        final float scale = context.getResources().getDisplayMetrics().density;
        return (int) (dpValue * scale + 0.5f);
    }
}
