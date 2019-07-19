//Author: samylee
//Date: 2019/07/19
//Blog Address: https://blog.csdn.net/samylee

package com.Main;

import java.util.ArrayList;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class Mtcnn {
	public Mtcnn(int row, int col) {
		float minl = row > col ? col : row;
		int MIN_DET_SIZE = 12;
		int minsize = 60;
		float m = (float) MIN_DET_SIZE / minsize;
		minl *= m;
		float factor = 0.709F;
		int factor_count = 0;

		while (minl > MIN_DET_SIZE) {
			if (factor_count > 0) {
				m = m * factor;
			}
			scales_.add(m);
			minl *= factor;
			factor_count++;
		}
		for (int i = 0; i < scales_.size(); i++) {
			Pnet pnet_ = new Pnet();
			simpleFace_.add(pnet_);
		}
	}

	public ArrayList<FaceInfo> findFace(Mat image) {
		// inital bbox
		firstBbox_.clear();
		firstOrderScore_.clear();
		secondBbox_.clear();
		secondBboxScore_.clear();
		thirdBbox_.clear();
		thirdBboxScore_.clear();
		
		// initial output face
		ArrayList<FaceInfo> faces = new ArrayList<FaceInfo>();

		int count = 0;
		for (int i = 0; i < scales_.size(); i++) {
			int changedH = (int) Math.ceil(image.rows() * scales_.get(i));
			int changedW = (int) Math.ceil(image.cols() * scales_.get(i));
			Imgproc.resize(image, reImage, new Size(changedW, changedH));
			simpleFace_.get(i).run(reImage, scales_.get(i));
			NetWork.nms(simpleFace_.get(i).boundingBox_, simpleFace_.get(i).bboxScore_,
					simpleFace_.get(i).nms_threshold, "Union");
			for (int k = 0; k < simpleFace_.get(i).boundingBox_.size(); k++) {
				if (simpleFace_.get(i).boundingBox_.get(k).exist) {
					firstBbox_.add(simpleFace_.get(i).boundingBox_.get(k));
					OrderScore order = new OrderScore();
					order.score = simpleFace_.get(i).boundingBox_.get(k).score;
					order.oriOrder = count;
					firstOrderScore_.add(order);
					count++;
				}
			}
			simpleFace_.get(i).bboxScore_.clear();
			simpleFace_.get(i).boundingBox_.clear();
		}
		// the first stage's nms
		if (count < 1) {
			return faces;
		}
		NetWork.nms(firstBbox_, firstOrderScore_, nms_threshold[0], "Union");
		NetWork.refineAndSquareBbox(firstBbox_, image.rows(), image.cols());

		// second stage
		count = 0;
		for (int i = 0; i < firstBbox_.size(); i++) {
			if (firstBbox_.get(i).exist) {
				Rect temp = new Rect(firstBbox_.get(i).y1, firstBbox_.get(i).x1,
						firstBbox_.get(i).y2 - firstBbox_.get(i).y1, firstBbox_.get(i).x2 - firstBbox_.get(i).x1);
				Mat secImage = new Mat();
				Mat rectImage = new Mat(image, temp);
				Imgproc.resize(rectImage, secImage, new Size(24, 24));
				refineNet.run(secImage);
				if (refineNet.score_.pdata.get(1) > refineNet.Rthreshold) {
					for (int k = 0; k < 4; k++) {
						firstBbox_.get(i).regreCoord[k] = refineNet.location_.pdata.get(k);
					}
					firstBbox_.get(i).area = (firstBbox_.get(i).x2 - firstBbox_.get(i).x1)
							* (firstBbox_.get(i).y2 - firstBbox_.get(i).y1);
					firstBbox_.get(i).score = refineNet.score_.pdata.get(1);
					secondBbox_.add(firstBbox_.get(i));
					OrderScore order = new OrderScore();
					order.score = firstBbox_.get(i).score;
					order.oriOrder = count++;
					secondBboxScore_.add(order);
				} else {
					firstBbox_.get(i).exist = false;
				}
			}
		}
		if (count < 1) {
			return faces;
		}
		NetWork.nms(secondBbox_, secondBboxScore_, nms_threshold[1], "Union");
		NetWork.refineAndSquareBbox(secondBbox_, image.rows(), image.cols());

		// third stage
		count = 0;
		for (int i = 0; i < secondBbox_.size(); i++) {
			if (secondBbox_.get(i).exist) {
				Rect temp = new Rect(secondBbox_.get(i).y1, secondBbox_.get(i).x1,
						secondBbox_.get(i).y2 - secondBbox_.get(i).y1, secondBbox_.get(i).x2 - secondBbox_.get(i).x1);
				Mat thirdImage = new Mat();
				Mat rectImage = new Mat(image, temp);
				Imgproc.resize(rectImage, thirdImage, new Size(48, 48));
				outNet.run(thirdImage);
				ArrayList<Float> pp = null;
				if (outNet.score_.pdata.get(1) > outNet.Othreshold) {
					for (int k = 0; k < 4; k++) {
						secondBbox_.get(i).regreCoord[k] = outNet.location_.pdata.get(k);
					}
					secondBbox_.get(i).area = (secondBbox_.get(i).x2 - secondBbox_.get(i).x1)
							* (secondBbox_.get(i).y2 - secondBbox_.get(i).y1);
					secondBbox_.get(i).score = outNet.score_.pdata.get(1);

					pp = outNet.keyPoint_.pdata;
					for (int num = 0; num < 5; num++) {
						secondBbox_.get(i).ppoint[num] = secondBbox_.get(i).y1
								+ (secondBbox_.get(i).y2 - secondBbox_.get(i).y1) * (pp.get(num));
					}
					for (int num = 0; num < 5; num++) {
						secondBbox_.get(i).ppoint[num + 5] = secondBbox_.get(i).x1
								+ (secondBbox_.get(i).x2 - secondBbox_.get(i).x1) * (pp.get(num + 5));
					}

					thirdBbox_.add(secondBbox_.get(i));
					OrderScore order = new OrderScore();
					order.score = secondBbox_.get(i).score;
					order.oriOrder = count++;
					thirdBboxScore_.add(order);
				} else {
					secondBbox_.get(i).exist = false;
				}
			}
		}
		if (count < 1) {
			return faces;
		}
		NetWork.refineAndSquareBbox(thirdBbox_, image.rows(), image.cols());
		NetWork.nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");

		for (int i = 0; i < thirdBbox_.size(); i++) {
			if (thirdBbox_.get(i).exist) {
				FaceInfo faceInfo = new FaceInfo();

				// face rect
				faceInfo.faceRect = new Rect(thirdBbox_.get(i).y1, thirdBbox_.get(i).x1,
						thirdBbox_.get(i).y2 - thirdBbox_.get(i).y1 + 1,
						thirdBbox_.get(i).x2 - thirdBbox_.get(i).x1 + 1);

				// face keypoint
				faceInfo.keyPoints = new ArrayList<Float>();
				for (int num = 0; num < 10; num++) {
					faceInfo.keyPoints.add(thirdBbox_.get(i).ppoint[num]);
				}

				faces.add(faceInfo);
			}
		}

		return faces;
	}

	private Mat reImage = new Mat();
	private float[] nms_threshold = { 0.7F, 0.7F, 0.7F };
	private ArrayList<Float> scales_ = new ArrayList<Float>();

	private ArrayList<Pnet> simpleFace_ = new ArrayList<Pnet>();
	private ArrayList<Bbox> firstBbox_ = new ArrayList<Bbox>();
	private ArrayList<OrderScore> firstOrderScore_ = new ArrayList<OrderScore>();

	private Rnet refineNet = new Rnet();
	private ArrayList<Bbox> secondBbox_ = new ArrayList<Bbox>();
	private ArrayList<OrderScore> secondBboxScore_ = new ArrayList<OrderScore>();

	private Onet outNet = new Onet();
	private ArrayList<Bbox> thirdBbox_ = new ArrayList<Bbox>();
	private ArrayList<OrderScore> thirdBboxScore_ = new ArrayList<OrderScore>();
}