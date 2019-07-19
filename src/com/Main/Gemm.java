//Author: samylee
//Date: 2019/07/19
//Blog Address: https://blog.csdn.net/samylee

package com.Main;

import java.util.ArrayList;

public class Gemm {
	static void gemm_cpu(int M, int N, int K, float ALPHA, ArrayList<Float> A, int lda, ArrayList<Float> B, int ldb,
			float BETA, ArrayList<Float> C, int ldc) {
		if (BETA != 1) {
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					int idxCount = i * ldc + j;
					C.set(idxCount, C.get(idxCount) * BETA);
				}
			}
		}
		for (int t = 0; t < M; ++t) {
			for (int j = 0; j < N; ++j) {
				float sum = 0;
				for (int k = 0; k < K; ++k) {
					sum += ALPHA * A.get(t * lda + k) * B.get(j * ldb + k);
				}
				C.set(t * ldc + j, C.get(t * ldc + j) + sum);
			}
		}
	}

	static void gemv_cpu(int M, int N, float ALPHA, ArrayList<Float> A, int lda, ArrayList<Float> X, int ldb,
			float BETA, ArrayList<Float> Y, int ldc) {
		for (int i = 0; i < M; i++) {
			float sum = 0;
			for (int j = 0; j < N; j++) {
				sum += A.get(i * lda + j) * X.get(j * ldb);
			}
			Y.set(i, sum);
		}
	}
}
