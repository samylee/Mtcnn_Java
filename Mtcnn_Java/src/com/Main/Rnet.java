//Author: samylee
//Date: 2019/07/19
//Blog Address: https://blog.csdn.net/samylee

package com.Main;

import java.util.ArrayList;

import org.opencv.core.Mat;

public class Rnet {
	public Rnet() {
		Rthreshold = 0.7F;

		this.rgb = new PBox();
		this.conv1_matrix = new PBox();
		this.conv1_out = new PBox();
		this.pooling1_out = new PBox();

		this.conv2_matrix = new PBox();
		this.conv2_out = new PBox();
		this.pooling2_out = new PBox();

		this.conv3_matrix = new PBox();
		this.conv3_out = new PBox();

		this.fc4_out = new PBox();

		this.score_ = new PBox();
		this.location_ = new PBox();

		this.conv1_wb = new Weight();
		this.prelu_gmma1 = new PRelu();
		this.conv2_wb = new Weight();
		this.prelu_gmma2 = new PRelu();
		this.conv3_wb = new Weight();
		this.prelu_gmma3 = new PRelu();
		this.fc4_wb = new Weight();
		this.prelu_gmma4 = new PRelu();
		this.score_wb = new Weight();
		this.location_wb = new Weight();
		// // w sc lc ks s p
		long conv1 = NetWork.initConvAndFc(this.conv1_wb, 28, 3, 3, 1, 0);
		NetWork.initpRelu(this.prelu_gmma1, 28);
		long conv2 = NetWork.initConvAndFc(this.conv2_wb, 48, 28, 3, 1, 0);
		NetWork.initpRelu(this.prelu_gmma2, 48);
		long conv3 = NetWork.initConvAndFc(this.conv3_wb, 64, 48, 2, 1, 0);
		NetWork.initpRelu(this.prelu_gmma3, 64);
		long fc4 = NetWork.initConvAndFc(this.fc4_wb, 128, 576, 1, 1, 0);
		NetWork.initpRelu(this.prelu_gmma4, 128);
		long score = NetWork.initConvAndFc(this.score_wb, 2, 128, 1, 1, 0);
		long location = NetWork.initConvAndFc(this.location_wb, 4, 128, 1, 1, 0);
		long[] dataNumber = { conv1, 28, 28, conv2, 48, 48, conv3, 64, 64, fc4, 128, 128, score, 2, location, 4 };

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
		pointTeam.add(this.fc4_wb.pdata);
		pointTeam.add(this.fc4_wb.pbias);
		pointTeam.add(this.prelu_gmma4.pdata);
		pointTeam.add(this.score_wb.pdata);
		pointTeam.add(this.score_wb.pbias);
		pointTeam.add(this.location_wb.pdata);
		pointTeam.add(this.location_wb.pbias);

		String filename = "Rnet.txt";
		NetWork.readData(filename, dataNumber, pointTeam);

		// Init the network
		RnetImage2MatrixInit(rgb);
		NetWork.feature2MatrixInit(this.rgb, this.conv1_matrix, this.conv1_wb);
		NetWork.convolutionInit(this.conv1_wb, this.rgb, this.conv1_out, this.conv1_matrix);
		NetWork.maxPoolingInit(this.conv1_out, this.pooling1_out, 3, 2);
		NetWork.feature2MatrixInit(this.pooling1_out, this.conv2_matrix, this.conv2_wb);
		NetWork.convolutionInit(this.conv2_wb, this.pooling1_out, this.conv2_out, this.conv2_matrix);
		NetWork.maxPoolingInit(this.conv2_out, this.pooling2_out, 3, 2);
		NetWork.feature2MatrixInit(this.pooling2_out, this.conv3_matrix, this.conv3_wb);
		NetWork.convolutionInit(this.conv3_wb, this.pooling2_out, this.conv3_out, this.conv3_matrix);
		NetWork.fullconnectInit(this.fc4_wb, this.fc4_out);
		NetWork.fullconnectInit(this.score_wb, this.score_);
		NetWork.fullconnectInit(this.location_wb, this.location_);
	}

	void run(Mat image) {
		NetWork.image2Matrix(image, this.rgb);

		NetWork.feature2Matrix(this.rgb, this.conv1_matrix, this.conv1_wb);
		NetWork.convolution(this.conv1_wb, this.rgb, this.conv1_out, this.conv1_matrix);
		NetWork.prelu(this.conv1_out, this.conv1_wb.pbias, this.prelu_gmma1.pdata);

		NetWork.maxPooling(this.conv1_out, this.pooling1_out, 3, 2);

		NetWork.feature2Matrix(this.pooling1_out, this.conv2_matrix, this.conv2_wb);
		NetWork.convolution(this.conv2_wb, this.pooling1_out, this.conv2_out, this.conv2_matrix);
		NetWork.prelu(this.conv2_out, this.conv2_wb.pbias, this.prelu_gmma2.pdata);
		NetWork.maxPooling(this.conv2_out, this.pooling2_out, 3, 2);

		// conv3
		NetWork.feature2Matrix(this.pooling2_out, this.conv3_matrix, this.conv3_wb);
		NetWork.convolution(this.conv3_wb, this.pooling2_out, this.conv3_out, this.conv3_matrix);
		NetWork.prelu(this.conv3_out, this.conv3_wb.pbias, this.prelu_gmma3.pdata);

		// flatten
		NetWork.fullconnect(this.fc4_wb, this.conv3_out, this.fc4_out);
		NetWork.prelu(this.fc4_out, this.fc4_wb.pbias, this.prelu_gmma4.pdata);

		// conv51 score
		NetWork.fullconnect(this.score_wb, this.fc4_out, this.score_);
		NetWork.addbias(this.score_, this.score_wb.pbias);
		NetWork.softmax(this.score_);

		// conv5_2 location
		NetWork.fullconnect(this.location_wb, this.fc4_out, this.location_);
		NetWork.addbias(this.location_, this.location_wb.pbias);
		// pBoxShow(location_);
	}

	float Rthreshold;
	PBox score_;
	PBox location_;

	private PBox rgb;

	private PBox conv1_matrix;
	private PBox conv1_out;
	private PBox pooling1_out;

	private PBox conv2_matrix;
	private PBox conv2_out;
	private PBox pooling2_out;

	private PBox conv3_matrix;
	private PBox conv3_out;

	private PBox fc4_out;

	// Weight
	private Weight conv1_wb;
	private PRelu prelu_gmma1;
	private Weight conv2_wb;
	private PRelu prelu_gmma2;
	private Weight conv3_wb;
	private PRelu prelu_gmma3;
	private Weight fc4_wb;
	private PRelu prelu_gmma4;
	private Weight score_wb;
	private Weight location_wb;

	private void RnetImage2MatrixInit(PBox pbox) {
		pbox.channel = 3;
		pbox.height = 24;
		pbox.width = 24;

		long byteLenght = pbox.channel * pbox.height * pbox.width;
		pbox.pdata = new ArrayList<Float>();
		if (pbox.pdata == null) {
			System.out.println("the image2MatrixInit is failed!!");
		}
		for (int i = 0; i < byteLenght; i++) {
			pbox.pdata.add(0.0F);
		}
	}
}
