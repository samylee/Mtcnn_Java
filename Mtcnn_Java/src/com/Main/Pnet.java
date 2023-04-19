//Author: samylee
//Date: 2019/07/19
//Blog Address: https://blog.csdn.net/samylee

package com.Main;

import java.util.ArrayList;

import org.opencv.core.Mat;

public class Pnet {
	public Pnet() {
		Pthreshold = 0.6F;
		nms_threshold = 0.5F;
		firstFlag = true;
		this.rgb = new PBox();

		this.conv1_matrix = new PBox();
		this.conv1 = new PBox();
		this.maxPooling1 = new PBox();

		this.maxPooling_matrix = new PBox();
		this.conv2 = new PBox();

		this.conv3_matrix = new PBox();
		this.conv3 = new PBox();

		this.score_matrix = new PBox();
		this.score_ = new PBox();

		this.location_matrix = new PBox();
		this.location_ = new PBox();

		this.conv1_wb = new Weight();
		this.prelu_gmma1 = new PRelu();
		this.conv2_wb = new Weight();
		this.prelu_gmma2 = new PRelu();
		this.conv3_wb = new Weight();
		this.prelu_gmma3 = new PRelu();
		this.conv4c1_wb = new Weight();
		this.conv4c2_wb = new Weight();
		// w sc lc ks s p
		long conv1 = NetWork.initConvAndFc(this.conv1_wb, 10, 3, 3, 1, 0);
		NetWork.initpRelu(this.prelu_gmma1, 10);
		long conv2 = NetWork.initConvAndFc(this.conv2_wb, 16, 10, 3, 1, 0);
		NetWork.initpRelu(this.prelu_gmma2, 16);
		long conv3 = NetWork.initConvAndFc(this.conv3_wb, 32, 16, 3, 1, 0);
		NetWork.initpRelu(this.prelu_gmma3, 32);
		long conv4c1 = NetWork.initConvAndFc(this.conv4c1_wb, 2, 32, 1, 1, 0);
		long conv4c2 = NetWork.initConvAndFc(this.conv4c2_wb, 4, 32, 1, 1, 0);
		long[] dataNumber = { conv1, 10, 10, conv2, 16, 16, conv3, 32, 32, conv4c1, 2, conv4c2, 4 };

		ArrayList<ArrayList<Float>> pointTeam = new ArrayList<ArrayList<Float>>();
		pointTeam.add(this.conv1_wb.pdata);
		pointTeam.add(this.conv1_wb.pbias);
		pointTeam.add(this.prelu_gmma1.pdata);
		pointTeam.add(this.conv2_wb.pdata);
		pointTeam.add(this.conv2_wb.pbias);
		pointTeam.add(this.prelu_gmma2.pdata);
		pointTeam.add(this.conv3_wb.pdata);
		pointTeam.add(this.conv3_wb.pbias);
		pointTeam.add(this.prelu_gmma3.pdata);
		pointTeam.add(this.conv4c1_wb.pdata);
		pointTeam.add(this.conv4c1_wb.pbias);
		pointTeam.add(this.conv4c2_wb.pdata);
		pointTeam.add(this.conv4c2_wb.pbias);

		String filename = "Pnet.txt";
		NetWork.readData(filename, dataNumber, pointTeam);
	}

	void run(Mat image, float scale) {
		if (firstFlag) {
			NetWork.image2MatrixInit(image, this.rgb);

			NetWork.feature2MatrixInit(this.rgb, this.conv1_matrix, this.conv1_wb);
			NetWork.convolutionInit(this.conv1_wb, this.rgb, this.conv1, this.conv1_matrix);

			NetWork.maxPoolingInit(this.conv1, this.maxPooling1, 2, 2);
			NetWork.feature2MatrixInit(this.maxPooling1, this.maxPooling_matrix, this.conv2_wb);
			NetWork.convolutionInit(this.conv2_wb, this.maxPooling1, this.conv2, this.maxPooling_matrix);

			NetWork.feature2MatrixInit(this.conv2, this.conv3_matrix, this.conv3_wb);
			NetWork.convolutionInit(this.conv3_wb, this.conv2, this.conv3, this.conv3_matrix);

			NetWork.feature2MatrixInit(this.conv3, this.score_matrix, this.conv4c1_wb);
			NetWork.convolutionInit(this.conv4c1_wb, this.conv3, this.score_, this.score_matrix);

			NetWork.feature2MatrixInit(this.conv3, this.location_matrix, this.conv4c2_wb);
			NetWork.convolutionInit(this.conv4c2_wb, this.conv3, this.location_, this.location_matrix);
			firstFlag = false;
		}

		NetWork.image2Matrix(image, this.rgb);

		NetWork.feature2Matrix(this.rgb, this.conv1_matrix, this.conv1_wb);
		NetWork.convolution(this.conv1_wb, this.rgb, this.conv1, this.conv1_matrix);
		NetWork.prelu(this.conv1, this.conv1_wb.pbias, this.prelu_gmma1.pdata);
		// Pooling layer
		NetWork.maxPooling(this.conv1, this.maxPooling1, 2, 2);

		NetWork.feature2Matrix(this.maxPooling1, this.maxPooling_matrix, this.conv2_wb);
		NetWork.convolution(this.conv2_wb, this.maxPooling1, this.conv2, this.maxPooling_matrix);
		NetWork.prelu(this.conv2, this.conv2_wb.pbias, this.prelu_gmma2.pdata);
		// conv3
		NetWork.feature2Matrix(this.conv2, this.conv3_matrix, this.conv3_wb);
		NetWork.convolution(this.conv3_wb, this.conv2, this.conv3, this.conv3_matrix);
		NetWork.prelu(this.conv3, this.conv3_wb.pbias, this.prelu_gmma3.pdata);
		// conv4c1 score
		NetWork.feature2Matrix(this.conv3, this.score_matrix, this.conv4c1_wb);
		NetWork.convolution(this.conv4c1_wb, this.conv3, this.score_, this.score_matrix);
		NetWork.addbias(this.score_, this.conv4c1_wb.pbias);
		NetWork.softmax(this.score_);

		// conv4c2 location
		NetWork.feature2Matrix(this.conv3, this.location_matrix, this.conv4c2_wb);
		NetWork.convolution(this.conv4c2_wb, this.conv3, this.location_, this.location_matrix);
		NetWork.addbias(this.location_, this.conv4c2_wb.pbias);
		// softmax layer
		generateBbox(this.score_, this.location_, scale);
	}

	float nms_threshold;
	float Pthreshold;
	boolean firstFlag;
	ArrayList<Bbox> boundingBox_ = new ArrayList<Bbox>();
	ArrayList<OrderScore> bboxScore_ = new ArrayList<OrderScore>();

	// the image for mxnet conv
	private PBox rgb;
	private PBox conv1_matrix;
	// the 1th layer's out conv
	private PBox conv1;
	private PBox maxPooling1;
	private PBox maxPooling_matrix;
	// the 3th layer's out
	private PBox conv2;
	private PBox conv3_matrix;
	// the 4th layer's out out
	private PBox conv3;
	private PBox score_matrix;
	// the 4th layer's out out
	private PBox score_;
	// the 4th layer's out out
	private PBox location_matrix;
	private PBox location_;

	// Weight
	private Weight conv1_wb;
	private PRelu prelu_gmma1;
	private Weight conv2_wb;
	private PRelu prelu_gmma2;
	private Weight conv3_wb;
	private PRelu prelu_gmma3;
	private Weight conv4c1_wb;
	private Weight conv4c2_wb;

	private void generateBbox(PBox score, PBox location, float scale) {
		// for pooling
		int stride = 2;
		int cellsize = 12;
		int count = 0;
		// score p
		int pCount = score.width * score.height;
		int plocalCount = 0;

		for (int row = 0; row < score.height; row++) {
			for (int col = 0; col < score.width; col++) {
				if (score.pdata.get(pCount) > Pthreshold) {
					Bbox bbox = new Bbox();
					OrderScore order = new OrderScore();

					bbox.score = score.pdata.get(pCount);
					order.score = score.pdata.get(pCount);
					order.oriOrder = count;
					bbox.x1 = Math.round((stride * row + 1) / scale);
					bbox.y1 = Math.round((stride * col + 1) / scale);
					bbox.x2 = Math.round((stride * row + 1 + cellsize) / scale);
					bbox.y2 = Math.round((stride * col + 1 + cellsize) / scale);
					bbox.exist = true;
					bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
					for (int channel = 0; channel < 4; channel++)
						bbox.regreCoord[channel] = location.pdata
								.get(plocalCount + channel * location.width * location.height);
					boundingBox_.add(bbox);
					bboxScore_.add(order);
					count++;
				}
				pCount++;
				plocalCount++;
			}
		}
	}
}