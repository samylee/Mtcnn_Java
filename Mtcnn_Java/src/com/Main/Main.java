//Author: samylee
//Date: 2019/07/19
//Blog Address: https://blog.csdn.net/samylee

package com.Main;

import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class Main {
	public static void main(String[] args) {
		// load OpenCV library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		String testType = "image";
		if (testType.equals("image")) {
			Mat image = Imgcodecs.imread("test.jpg");
			Mtcnn mtcnn = new Mtcnn(image.rows(), image.cols());
			ArrayList<FaceInfo> faces = mtcnn.findFace(image);
			Scalar scalar = new Scalar(0, 255, 0);

			// draw and show
			for (int i = 0; i < faces.size(); i++) {
				Point p1 = new Point(faces.get(i).faceRect.x, faces.get(i).faceRect.y);
				Point p2 = new Point(faces.get(i).faceRect.x + faces.get(i).faceRect.width - 1,
						faces.get(i).faceRect.y + faces.get(i).faceRect.height - 1);
				Imgproc.rectangle(image, p1, p2, scalar, 2);

				for (int num = 0; num < 5; num++) {
					Point kp = new Point(faces.get(i).keyPoints.get(num), faces.get(i).keyPoints.get(num + 5));
					Imgproc.circle(image, kp, 3, scalar, -1);
				}
			}

			HighGui.imshow("test", image);
			HighGui.waitKey(0);
		} else {
			// initial video capture
			VideoCapture cap = new VideoCapture(0);
			if (!cap.isOpened()) {
				System.out.println("can not open cam");
				return;
			}
			int imgW = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
			int imgH = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);

			Mtcnn mtcnn = new Mtcnn(imgH, imgW);
			Scalar scalar = new Scalar(0, 255, 0);
			Mat frame = new Mat();

			while (true) {
				cap.read(frame);
				if (frame.empty()) {
					System.out.println("can not read image");
					break;
				}

				ArrayList<FaceInfo> faces = mtcnn.findFace(frame);

				// draw and show
				for (int i = 0; i < faces.size(); i++) {
					Point p1 = new Point(faces.get(i).faceRect.x, faces.get(i).faceRect.y);
					Point p2 = new Point(faces.get(i).faceRect.x + faces.get(i).faceRect.width - 1,
							faces.get(i).faceRect.y + faces.get(i).faceRect.height - 1);
					Imgproc.rectangle(frame, p1, p2, scalar, 2);

					for (int num = 0; num < 5; num++) {
						Point kp = new Point(faces.get(i).keyPoints.get(num), faces.get(i).keyPoints.get(num + 5));
						Imgproc.circle(frame, kp, 3, scalar, -1);
					}
				}

				HighGui.imshow("test", frame);
				HighGui.waitKey(1);

			}
			// release
			cap.release();
			frame.release();
		}
	}
}
