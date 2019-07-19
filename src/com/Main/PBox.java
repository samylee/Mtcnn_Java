//Author: samylee
//Date: 2019/07/19
//Blog Address: https://blog.csdn.net/samylee

package com.Main;

import java.util.ArrayList;

import org.opencv.core.Rect;

public class PBox {
	ArrayList<Float> pdata;
	int width;
	int height;
	int channel;
}

class PRelu {
	ArrayList<Float> pdata;
	int width;
}

class Weight {
	ArrayList<Float> pdata;
	ArrayList<Float> pbias;
	int lastChannel;
	int selfChannel;
	int kernelSize;
	int stride;
	int pad;
};

class Bbox {
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
	float area;
	boolean exist;
	float[] ppoint = new float[10];
	float[] regreCoord = new float[4];
}

class OrderScore {
	float score;
	int oriOrder;
}

class FaceInfo {
	Rect faceRect;
	ArrayList<Float> keyPoints;
}
