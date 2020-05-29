package com.tencent.tnn.demo.common.component;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceView;

import com.tencent.tnn.demo.FaceDetector;

import java.util.ArrayList;


public class DrawView extends SurfaceView
{
    private static String TAG = DrawView.class.getSimpleName();
    private Paint paint = new Paint();
    private ArrayList<Rect> rects = new ArrayList<Rect>();


    public DrawView(Context context, AttributeSet attrs)
    {
        super(context, attrs);
        paint.setARGB(255, 0, 255, 0);
        paint.setStrokeWidth(3);
        paint.setStyle(Paint.Style.STROKE);
        setWillNotDraw(false);
    }


    public void addFaceRect(FaceDetector.FaceInfo[] facestatus, int w, int h)
    {
        rects.clear();
        Log.d(TAG, "canvas " + getWidth() + "x" + getHeight() + " wh " + w + "x" +h);
        float scalew = getWidth() / (float)w;
        float scaleh = getHeight() / (float)h;
        if (facestatus != null && facestatus.length!=0)
        {
            for (int i=0; i<facestatus.length; i++)
            {
                rects.add(new Rect((int)(facestatus[i].x1 * scalew), (int)(facestatus[i].y1 * scaleh), (int)(facestatus[i].x2 * scalew), (int)(facestatus[i].y2 * scaleh)));
            }
        }

        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas)
    {
        Log.d(TAG, "draw face count " + rects.size());
        if (rects.size() > 0)
        {
            for (int i=0; i<rects.size(); i++) {
                Log.d(TAG, "rect " + rects.get(i));
                paint.setARGB(255, 0, 255, 0);
                canvas.drawRect(rects.get(i), paint);
            }

        }
    }
}
