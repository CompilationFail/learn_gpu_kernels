#include "utils.cu"

namespace matmul{

constexpr int LN = 16, LM = 16, D = 128;
constexpr int LK = 32;
constexpr int RX = 4, RY = 4;
constexpr int KPART = (LM / RX) * (LN / RY);
constexpr int N_KPART = D / KPART;
// each thread handle [n,n+4),[m,m+4),[k,k+KPART)

// (N * K) * (M * K)^T
__global__ void matmul(const T *a, const T *b, T *C, int N, int M, int K) {
	// __shared__ T A[LN][LK], B[LM][LK], s[LN][LM][N_KPART];
	__shared__ T A[LK][LN], B[LK][LM], s[LN][LM][N_KPART];
	const int n = blockIdx.x * LN, m = blockIdx.y * LM, i0 = threadIdx.x;
	const int mix = i0 / LM, miy = i0 % LM;
	T ts[RX][RY];
	// loops will be unroll and ts will be put into register file
	#pragma unroll
	for(int i = 0; i < RX; ++i) 
		for(int j = 0; j < RY; ++j) 
			ts[i][j] = 0;
	const int u = i0 % (LN / RX) * RX;
	const int v = i0 / (LN / RX) % (LM / RY) * RY;
	const int kid = i0 / KPART;
	for(int k = 0; k < K; k += LK) {
		// TODO: no conflict right?
		/*int ix = i0 / LK, iy = i0 % LK;
		for(int i = ix; i < LN; i += D / LK) 
			A[i][iy] = a[(i + n) * K + k + iy];*/
		int ix = i0 / (LK / 4), iy = i0 % (LK / 4);
		for(int i = ix; i < LN; i += D / (LK / 4)) 
			copy4_stride_dst<LN+1>(&a[(i + n) * K + k + iy * 4], &A[iy * 4][i]);
		/*ix = i0 / LM, iy = i0 % LM;
		for(int i = ix; i < LK; i += D / LM) 
			B[iy][i] = b[(k + i) * M + (iy + m)];*/
		ix = i0 / (LM / 4), iy = i0 % (LM / 4);
		for(int i = ix; i < LK; i += D / (LM / 4)) {
			copy4(&b[(k + i) * M + m + iy * 4], &B[i][iy*4]);
		}
		__syncthreads();
		for(int w = kid; w < LK; w += N_KPART) {
			T a[RX], b[RY];
			#pragma unroll
			for(int i = 0; i < RX; i += 4) {
				copy4(&A[w][u + i], a + i);
			}
			#pragma unroll
			for(int i = 0; i < RY; i += 4) {
				copy4(&B[w][v + i], b + i);
			}
			#pragma unroll
			for(int i = 0; i < RX; ++i) 
				for(int j = 0; j < RY; ++j) {
					ts[i][j] += a[i] * b[j];
				}
		}
		/*for(int i = mix; i < LN; i += D / LM) {
			int j = miy;
			T t = 0;
			for(int k = 0; k < LK; ++k) 
				t += A[i][k] * B[j][k];
			s[i][j] += t;
		}*/
	}
	#pragma unroll
	for(int i = 0; i < RX; ++i) {
		for(int j = 0; j < RY; j ++) {
			s[u + i][v + j][kid] = ts[i][j];
		}
	}
	__syncthreads();
	for(int i = mix; i < LN; i += D / LM) {
		int j = miy;
		T t = 0;
		for(int k = 0; k < N_KPART; k += 4) {
			T tmp[4];
			copy4(&s[i][j][k], tmp);
			for(int w = 0; w < 4; ++w) t += tmp[w];
		}
		C[(i + n) * M + j + m] = t;
	}
}

int test_matmul(bool verify, int N, int M, int K) { 
	auto last = std::chrono::high_resolution_clock::now();
	auto X = Tensor<T>(N * M).rdrange(1,10).todevice();
	auto Y = Tensor<T>(M * K).rdrange(1,10).todevice();
	auto Z = Tensor<T>(N * K);
	auto Y_T = Tensor<T>(M * K, 0, 1);
	measurel("malloc & cudaMemcpy")

	float c2=0;
	int cases = 500;
	for(int cas = 0; cas < cases; ++cas) {
		matmul<<<dim3(N / LN, K / LM), D>>>(X.d(), Y.d(), Z.d(), N, K, M);
		cudaDeviceSynchronize();
	}
	measurec("matmul kernel", c2);
	printf("matmul kernel average: %.3lfms\n", c2 / cases);
	Z.tohost();
	checkCudaFail();
	int flag = 1;
	if(verify) {
		auto x = (T (*)[M])X.h();
		auto y = (T (*)[K])Y.h();
		auto z = (T (*)[K])Z.h();
		for(int i = 0; i < N; ++i) {
			for(int j = 0; j < K; ++j) {
				T s = 0;
				for(int k = 0; k < M; ++k) {
					s += x[i][k] * y[k][j];
				}
				if(!fp_equal(s, z[i][j])) {
					flag = 0;
					printf("i=%d,j=%d,z=%.3lf jury=%.3lf\n", i,j,z[i][j],s);
				}
			}
		}
	}
	measurel("check")
	return !flag;
}
}
using matmul::test_matmul;
