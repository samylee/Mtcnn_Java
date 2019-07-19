//Author: samylee
//Date: 2019/07/19
//Blog Address: https://blog.csdn.net/samylee

package com.Main;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class NetWork {
	static void addbias(PBox pbox, ArrayList<Float> pbias) {
		if (pbox.pdata == null) {
			System.out.println("Relu feature is NULL!!");
			return;
		}
		if (pbias == null) {
			System.out.println("the  Relu bias is NULL!!");
			return;
		}
		int opCount = 0;
		int pbCount = 0;

		long dis = pbox.width * pbox.height;
		for (int channel = 0; channel < pbox.channel; channel++) {
			for (int col = 0; col < dis; col++) {
				pbox.pdata.set(opCount, pbox.pdata.get(opCount) + pbias.get(pbCount));
				opCount++;
			}
			pbCount++;
		}
	}

	static void maxPooling(PBox pbox, PBox Matrix, int kernelSize, int stride) {
		if (pbox.pdata == null) {
			System.out.println("the feature2Matrix pbox is NULL!!");
			return;
		}
		int pCount = 0;
		int pInCount = 0;
		int ptempCount = 0;
		float maxNum = 0;
		if ((pbox.width - kernelSize) % stride == 0 && (pbox.height - kernelSize) % stride == 0) {
			for (int row = 0; row < Matrix.height; row++) {
				for (int col = 0; col < Matrix.width; col++) {
					pInCount = row * stride * pbox.width + col * stride;
					for (int channel = 0; channel < pbox.channel; channel++) {
						ptempCount = pInCount + channel * pbox.height * pbox.width;
						maxNum = pbox.pdata.get(ptempCount);
						for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
							for (int i = 0; i < kernelSize; i++) {
								if (maxNum < pbox.pdata.get(ptempCount + i + kernelRow * pbox.width)) {
									maxNum = pbox.pdata.get(ptempCount + i + kernelRow * pbox.width);
								}
							}
						}
						Matrix.pdata.set(pCount + channel * Matrix.height * Matrix.width, maxNum);
					}
					pCount++;
				}
			}
		} else {
			int diffh = 0, diffw = 0;
			for (int channel = 0; channel < pbox.channel; channel++) {
				pInCount = channel * pbox.height * pbox.width;
				for (int row = 0; row < Matrix.height; row++) {
					for (int col = 0; col < Matrix.width; col++) {
						ptempCount = pInCount + row * stride * pbox.width + col * stride;
						maxNum = pbox.pdata.get(ptempCount);
						diffh = row * stride - pbox.height + 1;
						diffw = col * stride - pbox.height + 1;
						for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
							if ((kernelRow + diffh) > 0)
								break;
							for (int i = 0; i < kernelSize; i++) {
								if ((i + diffw) > 0)
									break;
								if (maxNum < pbox.pdata.get(ptempCount + i + kernelRow * pbox.width)) {
									maxNum = pbox.pdata.get(ptempCount + i + kernelRow * pbox.width);
								}
							}
						}
						Matrix.pdata.set(pCount++, maxNum);
					}
				}
			}
		}
	}

	static void maxPoolingInit(PBox pbox, PBox Matrix, int kernelSize, int stride) {
		Matrix.width = (int) Math.ceil((float) (pbox.width - kernelSize) / stride + 1);
		Matrix.height = (int) Math.ceil((float) (pbox.height - kernelSize) / stride + 1);
		Matrix.channel = pbox.channel;
		long byteLenght = Matrix.channel * Matrix.width * Matrix.height;
		Matrix.pdata = new ArrayList<Float>();
		if (Matrix.pdata == null) {
			System.out.println("neicun muyou shenqing chengong!!");
		}
		for (int i = 0; i < byteLenght; i++) {
			Matrix.pdata.add(0.0F);
		}
	}

	static void prelu(PBox pbox, ArrayList<Float> pbias, ArrayList<Float> prelu_gmma) {
		if (pbox.pdata == null) {
			System.out.println("the  Relu feature is NULL!!");
			return;
		}
		if (pbias == null) {
			System.out.println("the  Relu bias is NULL!!");
			return;
		}

		int opCount = 0;
		int pbCount = 0;
		int pgCount = 0;
		long dis = pbox.width * pbox.height;
		for (int channel = 0; channel < pbox.channel; channel++) {
			for (int col = 0; col < dis; col++) {
				pbox.pdata.set(opCount, pbox.pdata.get(opCount) + pbias.get(pbCount));
				if (pbox.pdata.get(opCount) > 0) {
					pbox.pdata.set(opCount, pbox.pdata.get(opCount));
				} else {
					pbox.pdata.set(opCount, pbox.pdata.get(opCount) * prelu_gmma.get(pgCount));
				}
				opCount++;
			}
			pbCount++;
			pgCount++;
		}
	}

	static void fullconnect(Weight weight, PBox pbox, PBox outpBox) {
		if (pbox.pdata == null) {
			System.out.println("the fc feature is NULL!!");
			return;
		}
		if (weight.pdata == null) {
			System.out.println("the fc weight is NULL!!");
			return;
		}

		Gemm.gemv_cpu(weight.selfChannel, weight.lastChannel, 1, weight.pdata, weight.lastChannel, pbox.pdata, 1, 0,
				outpBox.pdata, 1);
	}

	static void fullconnectInit(Weight weight, PBox outpBox) {
		outpBox.channel = weight.selfChannel;
		outpBox.width = 1;
		outpBox.height = 1;

		long byteLenght = weight.selfChannel;
		outpBox.pdata = new ArrayList<Float>();
		if (outpBox.pdata == null) {
			System.out.println("the fullconnectInit is failed!!");
		}
		for (int i = 0; i < byteLenght; i++) {
			outpBox.pdata.add(0.0F);
		}
	}

	static void convolution(Weight weight, PBox pbox, PBox outpBox, PBox matrix) {
		if (pbox.pdata == null) {
			System.out.println("the feature is NULL!!");
			return;
		}
		if (weight.pdata == null) {
			System.out.println("the weight is NULL!!");
			return;
		}

		Gemm.gemm_cpu(weight.selfChannel, matrix.height, matrix.width, 1, weight.pdata, matrix.width, matrix.pdata,
				matrix.width, 0, outpBox.pdata, matrix.height);
	}

	static void convolutionInit(Weight weight, PBox pbox, PBox outpBox, PBox matrix) {
		outpBox.channel = weight.selfChannel;
		outpBox.width = (pbox.width - weight.kernelSize) / weight.stride + 1;
		outpBox.height = (pbox.height - weight.kernelSize) / weight.stride + 1;
		long byteLenght = weight.selfChannel * matrix.height;
		outpBox.pdata = new ArrayList<Float>();
		if (outpBox.pdata == null) {
			System.out.println("neicun muyou shenqing chengong!!");
		}
		for (int i = 0; i < byteLenght; i++) {
			outpBox.pdata.add(0.0F);
		}
	}

	static void feature2Matrix(PBox pbox, PBox Matrix, Weight weight) {
		if (pbox.pdata == null) {
			System.out.println("the feature2Matrix pbox is NULL!!");
			return;
		}
		int kernelSize = weight.kernelSize;
		int stride = weight.stride;
		int w_out = (pbox.width - kernelSize) / stride + 1;
		int h_out = (pbox.height - kernelSize) / stride + 1;

		int m_count = 0;
		int pIn_count = 0;
		int pTemp_count = 0;
		for (int row = 0; row < h_out; row++) {
			for (int col = 0; col < w_out; col++) {
				pIn_count = row * stride * pbox.width + col * stride;

				for (int channel = 0; channel < pbox.channel; channel++) {
					pTemp_count = pIn_count + channel * pbox.height * pbox.width;
					for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
						for (int n = 0; n < kernelSize; n++) {
							Matrix.pdata.set(n + m_count, pbox.pdata.get(pTemp_count + n));
						}
						m_count += kernelSize;
						pTemp_count += pbox.width;
					}
				}
			}
		}
	}

	static void feature2MatrixInit(PBox pbox, PBox Matrix, Weight weight) {

		int kernelSize = weight.kernelSize;
		int stride = weight.stride;
		int w_out = (pbox.width - kernelSize) / stride + 1;
		int h_out = (pbox.height - kernelSize) / stride + 1;
		Matrix.width = pbox.channel * kernelSize * kernelSize;
		Matrix.height = w_out * h_out;
		Matrix.channel = 1;
		long byteLenght = Matrix.width * Matrix.height;
		Matrix.pdata = new ArrayList<Float>();
		if (Matrix.pdata == null) {
			System.out.println("neicun muyou shenqing chengong!!");
		}
		for (int i = 0; i < byteLenght; i++) {
			Matrix.pdata.add(0.0F);
		}
	}

	static void image2Matrix(Mat image, PBox pbox) {
		if (image.empty() || (image.type() != CvType.CV_8UC3)) {
			System.out.println("image's type is wrong!!Please set CV_8UC3");
			return;
		}
		if (pbox.pdata == null) {
			return;
		}

		int p_count = 0;
		double[] pixel = new double[3];
		for (int rowI = 0; rowI < image.rows(); rowI++) {
			for (int colK = 0; colK < image.cols(); colK++) {
				pixel = image.get(rowI, colK).clone();
				pbox.pdata.set(p_count, (float) ((pixel[0] - 127.5) * 0.0078125));
				pbox.pdata.set(p_count + image.rows() * image.cols(), (float) ((pixel[1] - 127.5) * 0.0078125));
				pbox.pdata.set(p_count + 2 * image.rows() * image.cols(), (float) ((pixel[2] - 127.5) * 0.0078125));
				p_count++;
			}
		}
	}

	static void image2MatrixInit(Mat image, PBox pbox) {
		if (image.empty() || (image.type() != CvType.CV_8UC3)) {
			System.out.println("image's type is wrong!!Please set CV_8UC3");
			return;
		}
		pbox.channel = image.channels();
		pbox.height = image.rows();
		pbox.width = image.cols();
		long byteLenght = pbox.channel * pbox.height * pbox.width;
		pbox.pdata = new ArrayList<Float>();
		if (pbox.pdata == null) {
			System.out.println("neicun muyou shenqing chengong!!");
		}
		for (int i = 0; i < byteLenght; i++) {
			pbox.pdata.add(0.0F);
		}
	}

	static long initConvAndFc(Weight weight, int schannel, int lchannel, int kersize, int stride, int pad) {
		weight.selfChannel = schannel;
		weight.lastChannel = lchannel;
		weight.kernelSize = kersize;
		weight.stride = stride;
		weight.pad = pad;

		// initial pbias
		weight.pbias = new ArrayList<Float>();
		if (weight.pbias == null) {
			System.out.println("neicun muyou shenqing chengong!!");
		}
		for (int i = 0; i < schannel; i++) {
			weight.pbias.add(0.0F);
		}

		// initial pdata
		long byteLenght = weight.selfChannel * weight.lastChannel * weight.kernelSize * weight.kernelSize;
		weight.pdata = new ArrayList<Float>();
		if (weight.pdata == null) {
			System.out.println("neicun muyou shenqing chengong!!");
		}
		for (int i = 0; i < byteLenght; i++) {
			weight.pdata.add(0.0F);
		}

		return byteLenght;
	}

	static void initpRelu(PRelu prelu, int width) {

		prelu.width = width;
		prelu.pdata = new ArrayList<Float>();
		if (prelu.pdata == null) {
			System.out.println("neicun muyou shenqing chengong!!");
		}
		for (int i = 0; i < width; i++) {
			prelu.pdata.add(0.0F);
		}
	}

	static void softmax(PBox pbox) {
		if (pbox.pdata == null) {
			System.out.println("the softmax's pdata is NULL , Please check !");
			return;
		}
		long p2DCount = 0;
		long p3DCount = 0;
		long mapSize = pbox.width * pbox.height;
		float eleSum = 0;
		for (int row = 0; row < pbox.height; row++) {
			for (int col = 0; col < pbox.width; col++) {
				eleSum = 0;
				for (int channel = 0; channel < pbox.channel; channel++) {
					p3DCount = p2DCount + channel * mapSize;
					pbox.pdata.set((int) p3DCount, (float) Math.exp(pbox.pdata.get((int) p3DCount)));
					eleSum += pbox.pdata.get((int) p3DCount);
				}
				for (int channel = 0; channel < pbox.channel; channel++) {
					p3DCount = p2DCount + channel * mapSize;
					pbox.pdata.set((int) p3DCount, pbox.pdata.get((int) p3DCount) / eleSum);
				}
				p2DCount++;
			}
		}
	}

	static void readData(String filename, long dataNumber[], ArrayList<ArrayList<Float>> pTeam) {
		BufferedReader in = null;
		try {
			in = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		String line;
		if (in != null) {
			int i = 0;
			int count = 0;
			int pTeam_count = 0;
			try {
				while ((line = in.readLine()) != null) {
					if (i < dataNumber[count]) {
						String newLine = line.substring(1, line.length() - 1);
						float data = Float.parseFloat(newLine);
						// *(pTeam[count])++ = atof(line.data());
						pTeam.get(count).set(pTeam_count++, data);
					} else {
						count++;
						dataNumber[count] += dataNumber[count - 1];
						pTeam_count = 0;

						String newLine = line.substring(1, line.length() - 1);
						float data = Float.parseFloat(newLine);
						// *(pTeam[count])++ = atof(line.data());
						pTeam.get(count).set(pTeam_count++, data);
					}
					i++;
				}
			} catch (NumberFormatException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} finally {
				try {
					in.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		} else {
			System.out.println("no such file" + filename);
		}
	}

	static void nms(ArrayList<Bbox> boundingBox_, ArrayList<OrderScore> bboxScore_, float overlap_threshold,
			String modelname) {
		if (boundingBox_.isEmpty()) {
			return;
		}
		ArrayList<Integer> heros = new ArrayList<Integer>();
		// sort the score(small -> big)
		Collections.sort(bboxScore_, new Comparator<OrderScore>() {
			@Override
			public int compare(OrderScore lsh, OrderScore rsh) {
				if (lsh.score > rsh.score) {
					return 1;
				} else if (lsh.score < rsh.score) {
					return -1;
				} else {
					return 0;
				}
			}
		});

		int order = 0;
		float IOU = 0;
		float maxX = 0;
		float maxY = 0;
		float minX = 0;
		float minY = 0;
		while (bboxScore_.size() > 0) {
			order = bboxScore_.get(bboxScore_.size() - 1).oriOrder;
			bboxScore_.remove(bboxScore_.size() - 1);
			if (order < 0) {
				continue;
			}
			heros.add(order);
			boundingBox_.get(order).exist = false;// delete it

			for (int num = 0; num < boundingBox_.size(); num++) {
				if (boundingBox_.get(num).exist) {
					// the iou
					maxX = (boundingBox_.get(num).x1 > boundingBox_.get(order).x1) ? boundingBox_.get(num).x1
							: boundingBox_.get(order).x1;
					maxY = (boundingBox_.get(num).y1 > boundingBox_.get(order).y1) ? boundingBox_.get(num).y1
							: boundingBox_.get(order).y1;
					minX = (boundingBox_.get(num).x2 < boundingBox_.get(order).x2) ? boundingBox_.get(num).x2
							: boundingBox_.get(order).x2;
					minY = (boundingBox_.get(num).y2 < boundingBox_.get(order).y2) ? boundingBox_.get(num).y2
							: boundingBox_.get(order).y2;
					// maxX1 and maxY1 reuse
					maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
					maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
					// IOU reuse for the area of two bbox
					IOU = maxX * maxY;
					if (modelname.compareTo("Union") == 0) {
						IOU = IOU / (boundingBox_.get(num).area + boundingBox_.get(order).area - IOU);
					} else if (modelname.compareTo("Min") == 0) {
						IOU = IOU / ((boundingBox_.get(num).area < boundingBox_.get(order).area)
								? boundingBox_.get(num).area : boundingBox_.get(order).area);
					}
					if (IOU > overlap_threshold) {
						boundingBox_.get(num).exist = false;
						for (int k = 0; k < bboxScore_.size(); k++) {
							if (bboxScore_.get(k).oriOrder == num) {
								bboxScore_.get(k).oriOrder = -1;
								break;
							}
						}
					}
				}
			}
		}
		for (int i = 0; i < heros.size(); i++) {
			boundingBox_.get(heros.get(i)).exist = true;
		}
	}

	static void refineAndSquareBbox(ArrayList<Bbox> vecBbox, int height, int width) {
		if (vecBbox.isEmpty()) {
			System.out.println("Bbox is empty!!");
			return;
		}
		float bbw = 0, bbh = 0, maxSide = 0;
		float h = 0, w = 0;
		float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
		for (int i = 0; i < vecBbox.size(); i++) {
			if (vecBbox.get(i).exist) {
				bbh = vecBbox.get(i).x2 - vecBbox.get(i).x1 + 1;
				bbw = vecBbox.get(i).y2 - vecBbox.get(i).y1 + 1;
				x1 = vecBbox.get(i).x1 + vecBbox.get(i).regreCoord[1] * bbh;
				y1 = vecBbox.get(i).y1 + vecBbox.get(i).regreCoord[0] * bbw;
				x2 = vecBbox.get(i).x2 + vecBbox.get(i).regreCoord[3] * bbh;
				y2 = vecBbox.get(i).y2 + vecBbox.get(i).regreCoord[2] * bbw;

				h = x2 - x1 + 1;
				w = y2 - y1 + 1;

				maxSide = (h > w) ? h : w;
				x1 = x1 + h * 0.5F - maxSide * 0.5F;
				y1 = y1 + w * 0.5F - maxSide * 0.5F;
				vecBbox.get(i).x2 = Math.round(x1 + maxSide - 1);
				vecBbox.get(i).y2 = Math.round(y1 + maxSide - 1);
				vecBbox.get(i).x1 = Math.round(x1);
				vecBbox.get(i).y1 = Math.round(y1);

				// boundary check
				if (vecBbox.get(i).x1 < 0) {
					vecBbox.get(i).x1 = 0;
				}
				if (vecBbox.get(i).y1 < 0) {
					vecBbox.get(i).y1 = 0;
				}
				if (vecBbox.get(i).x2 > height) {
					vecBbox.get(i).x2 = height - 1;
				}
				if (vecBbox.get(i).y2 > width) {
					vecBbox.get(i).y2 = width - 1;
				}

				vecBbox.get(i).area = (vecBbox.get(i).x2 - vecBbox.get(i).x1) * (vecBbox.get(i).y2 - vecBbox.get(i).y1);
			}
		}
	}
}