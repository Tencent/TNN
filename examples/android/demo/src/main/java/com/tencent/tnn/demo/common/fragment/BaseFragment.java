package com.tencent.tnn.demo.common.fragment;

import android.app.Fragment;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;

import com.tencent.tnn.demo.R;
import com.tencent.tnn.demo.common.component.TitleBar;


public abstract class BaseFragment extends Fragment implements View.OnClickListener {

    private LinearLayout root;
    private TitleBar titleBar;
    private LayoutInflater mInflater;
    public boolean NpuEnable = false;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        mInflater = inflater;
        View v = inflater.inflate(R.layout.base_fragment_layout, container, false);
        root = (LinearLayout) v.findViewById(R.id.avd_contain);
        titleBar = $(R.id.avd_title_bar);
        setFragmentView();
        //  closeBackKey();
        return v;
    }

    public <T> T $(int id) {
        return (T) root.findViewById(id);
    }

    public <T extends View> T $$(int id) {
        View v = root.findViewById(id);
        v.setOnClickListener(this);
        return (T) v;
    }


    public abstract void setFragmentView();
    public abstract void openCamera();
    public abstract void startPreview(SurfaceHolder surfaceHolder);
    public abstract void closeCamera();

    public View setView(int layoutId) {
        View content = mInflater.inflate(layoutId, null);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
        content.setLayoutParams(lp);
        root.addView(content);
        return root;
    }

    public void setTitleGone() {
        titleBar.setVisibility(View.GONE);
    }

    public void setTitle(String title) {
        titleBar.setTitle(title);
    }

    public void setTitleOnly(String title) {
        titleBar.setTitleOnly(title);
    }


    public void setRightImage() {
        titleBar.setRightImge();
    }

    public void setRightImageSrc(int resId) {
        titleBar.setRightImageSrc(resId);
    }

    public void setRightText(String right) {
        titleBar.setRightText(right);
    }

    public void setClickListener(TitleBar.TitleBarClick clickListener) {

        if (clickListener != null) {
            titleBar.setClickListener(clickListener);
        }
    }

    @Override
    public void onClick(View view) {

    }


}
