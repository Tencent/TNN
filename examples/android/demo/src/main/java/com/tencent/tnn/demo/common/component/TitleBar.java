package com.tencent.tnn.demo.common.component;

import android.content.Context;
import android.content.res.TypedArray;
import android.util.AttributeSet;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.tencent.tnn.demo.R;

public class TitleBar extends RelativeLayout implements View.OnClickListener {

    private static final String TAG = TitleBar.class.getSimpleName();

    private TextView mLeftText;
    private ImageView mLeftImage;
    private TextView mRightText;
    private ImageView mRightImage;
    private TextView mTitle;

    public TitleBar(Context context) {
        super(context);
        initView();
    }

    public TitleBar(Context context, AttributeSet attrs) {
        super(context, attrs);
        initView();
        getAttrs(context, attrs);
    }

    public TitleBar(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        initView();
        getAttrs(context, attrs);
    }

    private void initView() {

        LayoutInflater layoutInflater = (LayoutInflater) getContext().getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View root = layoutInflater.inflate(R.layout.title_bar_layout, this);
        root.findViewById(R.id.avd_left_button).setOnClickListener(this);
        root.findViewById(R.id.avd_right_button).setOnClickListener(this);
        mLeftText = (TextView) root.findViewById(R.id.avd_left_text);
        mRightText = (TextView) root.findViewById(R.id.avd_right_text);
        mLeftImage = (ImageView) root.findViewById(R.id.avd_left_image);
        mRightImage = (ImageView) root.findViewById(R.id.avd_right_image);
        mTitle = (TextView) root.findViewById(R.id.avd_bar_title);
    }

    private void getAttrs(Context context, AttributeSet attrs) {
        if (context == null) {
            Log.e(TAG, "传入context为空");
            return;
        }
        TypedArray ta = context.obtainStyledAttributes(attrs, R.styleable.TitleBarAttr);
        String mLeftString = ta.getString(R.styleable.TitleBarAttr_left_text);
        String mRightString = ta.getString(R.styleable.TitleBarAttr_right_text);
        String mTitleString = ta.getString(R.styleable.TitleBarAttr_bar_title);

        boolean leftImageVisible = ta.getBoolean(R.styleable.TitleBarAttr_left_image_visible, true);
        if (!leftImageVisible) {
            mLeftImage.setVisibility(GONE);
        }

        if (mTitleString != null) {
            mTitle.setText(mTitleString);
        } else {
            mTitle.setVisibility(INVISIBLE);
        }

        int mLeftImageId = ta.getResourceId(R.styleable.TitleBarAttr_left_image, 0);

        boolean rightImageVisible = ta.getBoolean(R.styleable.TitleBarAttr_right_image_visible, false);
        if (rightImageVisible) {
            mRightImage.setVisibility(VISIBLE);
        } else {
            mRightImage.setVisibility(GONE);
        }

        if (mRightString != null) {
            mRightText.setVisibility(VISIBLE);
            mRightText.setText(mRightString);
        } else {
            mRightText.setVisibility(GONE);
        }

        if (mLeftString != null) {
            mLeftText.setVisibility(VISIBLE);
            mLeftText.setText(mLeftString);
        } else {
            mLeftText.setVisibility(INVISIBLE);
        }

        if (mLeftImageId != 0) {
            mLeftImage.setImageDrawable(getResources().getDrawable(mLeftImageId));
        }
        ta.recycle();
    }


    @Override
    public void onClick(View view) {
        if (view.getId() == R.id.avd_left_button) {
            if (mClick != null) {
                mClick.onLeftClick();
            }
        }

        if (view.getId() == R.id.avd_right_button) {
            if (mClick != null) {
                mClick.onRightClick();
            }
        }
    }

    public interface TitleBarClick {
        void onLeftClick();

        void onRightClick();
    }

    private TitleBarClick mClick;

    public void setClickListener(TitleBarClick click) {
        mClick = click;
    }


    public void setLeftText(String text) {
        mLeftText.setVisibility(VISIBLE);
        mLeftText.setText(text);
        mLeftImage.setVisibility(VISIBLE);
    }

    public void setTitle(String title) {
        mTitle.setVisibility(VISIBLE);
        mTitle.setText(title);
    }

    public void setTitleOnly(String title) {
        mTitle.setVisibility(VISIBLE);
        mTitle.setText(title);

        mLeftText.setVisibility(INVISIBLE);
        mRightText.setVisibility(INVISIBLE);
        mLeftImage.setVisibility(INVISIBLE);
        mRightImage.setVisibility(INVISIBLE);
    }

    public void setRightImge() {
        mRightImage.setVisibility(VISIBLE);
    }

    public void setRightImageSrc(int id) {
        mRightImage.setImageResource(id);
    }

    public void setRightText(String text) {
        mRightText.setVisibility(VISIBLE);
        mRightText.setText(text);
    }
}
