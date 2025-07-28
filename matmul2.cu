#include "utils.cu"

namespace matmul{

constexpr int LN = 16, LM = 8, D = 128;
constexpr int LK = 16;
constexpr int KPART = (LM / 4) * (LN / 4);
constexpr int N_KPART = D / KPART;
// each thread handle [n,n+4),[m,m+4),[k,k+KPART)

// (N * K) * (M * K)^T
__global__ void matmul(const T *a, const T *b, T *C, int N, int M, int K) {
	__shared__ T A[LN][LK], B[LM][LK], s[LN][LM][N_KPART];
	const int n = blockIdx.x * LN, m = blockIdx.y * LM, i0 = threadIdx.x;
	const int mix = i0 / LM, miy = i0 % LM;
	T s00 = 0, s01 = 0, s02 = 0, s03 = 0;
	T s10 = 0, s11 = 0, s12 = 0, s13 = 0;
	T s20 = 0, s21 = 0, s22 = 0, s23 = 0;
	T s30 = 0, s31 = 0, s32 = 0, s33 = 0;
	const int u = i0 % (LN / 4) * 4;
	const int v = i0 / (LN / 4) % (LM / 4) * 4;
	const int kid = i0 / KPART;
	for(int k = 0; k < K; k += LK) {
		// TODO: no conflict right?
		int ix = i0 / LK, iy = i0 % LK;
		for(int i = ix; i < LN; i += D / LK) 
			A[i][iy] = a[(i + n) * K + k + iy];
		ix = i0 / LM, iy = i0 % LM;
		for(int i = ix; i < LK; i += D / LM) 
			B[iy][i] = b[(k + i) * M + (iy + m)];
		__syncthreads();
		for(int w = kid; w < LK; w += N_KPART) {
			T a0 = A[u][w], a1 = A[u + 1][w], a2 = A[u + 2][w], a3 = A[u + 3][w];
			T b0 = B[v][w], b1 = B[v + 1][w], b2 = B[v + 2][w], b3 = B[v + 3][w];
			s00 += a0 * b0, s01 += a0 * b1, s02 += a0 * b2, s03 += a0 * b3;
			s10 += a1 * b0, s11 += a1 * b1, s12 += a1 * b2, s13 += a1 * b3;
			s20 += a2 * b0, s21 += a2 * b1, s22 += a2 * b2, s23 += a2 * b3;
			s30 += a3 * b0, s31 += a3 * b1, s32 += a3 * b2, s33 += a3 * b3;
		}
		/*for(int i = mix; i < LN; i += D / LM) {
			int j = miy;
			T t = 0;
			for(int k = 0; k < LK; ++k) 
				t += A[i][k] * B[j][k];
			s[i][j] += t;
		}*/
	}
	s[u  ][v][kid] = s00; s[u  ][v+1][kid] = s01; s[u  ][v+2][kid] = s02; s[u  ][v+3][kid] = s03;
	s[u+1][v][kid] = s10; s[u+1][v+1][kid] = s11; s[u+1][v+2][kid] = s12; s[u+1][v+3][kid] = s13;
	s[u+2][v][kid] = s20; s[u+2][v+1][kid] = s21; s[u+2][v+2][kid] = s22; s[u+2][v+3][kid] = s23;
	s[u+3][v][kid] = s30; s[u+3][v+1][kid] = s31; s[u+3][v+2][kid] = s32; s[u+3][v+3][kid] = s33;
	__syncthreads();
	for(int i = mix; i < LN; i += D / LM) {
		int j = miy;
		T t = 0;
		for(int k = 0; k < N_KPART; ++k) {
			t += s[i][j][k];
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
