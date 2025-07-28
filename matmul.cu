#include "utils.cu"

namespace matmul{

/*constexpr int LX = 128, LY = 32, DX = 8, DY = 16, THREADS = DX * DY;
__global__ void matmul(const T *x, const T *y, T *z, int N, int M, int K) {
	int i0 = threadIdx.x;
	int j0 = threadIdx.y;
	int ij0 = i0 * DY + j0;
	static constexpr int D = TransposeD;
	__shared__ T tmp[LY][LX], tx[LX], s[THREADS]; // [THREADS/4];
	int ibase = blockIdx.x * LX;
	int jbase = blockIdx.y * LY;
	// [ibase, ibase + LX), [jbase, jbase + LY) 
	// -> [jbase, jbase + LY), [ibase, ibase + LX)
	for(int i =  i0; i < LX; i += DX) {
		for(int j = j0; j < LY; j += DY) { 
			// tmp[j0][i0] = x[(i0 + ibase) * M + j0 + jbase];
			// swiss to avoid bank conflict
			tmp[j][(i + 2*j) % D] = y[(i + ibase) * K + j + jbase];
		}
	}
	__syncthreads();
	for(int i = 0; i < N; ++i) {
		// 读 x 的一行中 [ibase, ibase+LX) 的一段
		for(int j = ij0; j < LX; j += THREADS) 
			tx[j] = x[i * M + ibase + j];
		__syncthreads();
		for(int j = 0; j < LY; ++j) {
			s[ij0] = 0;
			for(int k = ij0; k < LX; k += THREADS) {
				s[ij0] += tx[k] * tmp[j][k];
			}
			__syncthreads();
			for(int stride = THREADS; stride /= 2;) {
				if(ij0 < stride) s[ij0] += s[ij0 + stride];
				__syncthreads();
			}
			// 这里用 reduce 会把计算复杂度变成 NK(M/threads + logM)
			// 看看如果变成瓶颈了就需要改
			if(ij0 == 0) z[i * K + j + jbase] += s[0];
		}
	}
}*/

/*__global__ void matmul(const T *x, const T *y, T *z, int N, int M, int K) {
	__shared__ T s[1024];
	const int a = blockIdx.x, b = blockIdx.y, i0 = threadIdx.x, D = blockDim.x;
	s[i0] = 0;
	const T *xbase = x + a * M;
	const T *ybase = y + b * M;
	for(int i = i0; i < M; i += D) {
		s[i0] += xbase[i] * ybase[i];
	}
	__syncthreads();
	for(int stride = D; stride /= 2;) {
		if(i0 < stride) s[i0] += s[i0 + stride];
		__syncthreads();
	}
	if(i0 == 0) z[a * K + b] = s[0];
}*/

constexpr int LX = 16, LY = 8, D = 128;
constexpr int MAXM = 512; // 

__global__ void matmul(const T *x, const T *y, T *z, int N, int M, int K) {
	__shared__ T tx[MAXM], ty[LY][MAXM], s[LY][D+1]; // D+1 to avoid bank conflict
	const int a = blockIdx.x * LX, b = blockIdx.y * LY, i0 = threadIdx.x;
	const T *xbase = x + a * M;
	// const T *ybase = y + b * M;
	for(int i = i0; i < M; i += D) {
		for(int v = 0; v < LY; ++v) {
			ty[v][i] = y[i * K + b + v]; //ybase[i];
			// ybase += M;
		}
	}
	__syncthreads();
	for(int u = 0;  u < LX; ++u) {
		for(int i = i0; i < M; i += D) {
			tx[i] = xbase[i];
		}
		xbase += M;
		for(int v = 0; v < LY; ++v) {
			s[v][i0] = 0;
		}
		for(int i = i0; i < M; i += D) {
			T x = tx[i];
			for(int v = 0; v < LY; ++v) {
				s[v][i0] += x * ty[v][i];
			}
		}
		__syncthreads();
		if(i0 < LY) {
			T ss = 0;
			for(int i = 0; i < D; ++i) ss += s[i0][i];
			z[(a + u) * K + (b + i0)] = ss;
		}
		__syncthreads();
	}
}

/*
	const int E = D / LY,  ex = i0 / E, ey = i0 % E;
	// 每行 8 个 worker一起进行求和
	for(int u = 0;  u < LX; ++u) {
		for(int i = i0; i < M; i += D) {
			tx[i] = xbase[i];
		}
		xbase += M;
		for(int v = 0; v < LY; ++v) {
			s[v][i0] = 0;
			for(int i = i0; i < M; i += D) {
				s[v][i0] += tx[i] * ty[v][i];
			}
		}
		__syncthreads();
		for(int i = ey + E; i < D; i += E) s[ex][ey] += s[ex][i];
		__syncthreads();
		if(i0 < LY) {
			T ss = 0;
			for(int i = 0; i < E; i++) ss += s[i0][i];
			z[(a + u) * K + (b + i0)] = ss;
		}
		__syncthreads();
	}
*/

template <class T>
void matmul_wrapper(Tensor <T>&X, Tensor<T> &Y, Tensor<T> &Z, int N, int M, int K) {
	// 先转置再做会更快吗？
	/*auto Y_T = Tensor<T>(M * K, 0, 1);
	transpose_wrapper(Y, Y_T, M, K);
	y2.tohost();
	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < K; ++j) {
			if(!fp_equal(Y.h()[i * K + j], Y_T.h()[j * M + i])) {
				assert(0);
			}
		}
	}*/
	assert(N % LX == 0);
	assert(N % LY == 0);
	assert(M <= MAXM);
	matmul<<<dim3(N / LX, K / LY), D>>>(X.d(), Y.d(), Z.d(), N, M, K);
	cudaDeviceSynchronize();
}

int test_matmul(bool verify, int N, int M, int K) { 
	auto last = std::chrono::high_resolution_clock::now();
	auto X = Tensor<T>(N * M).rdrange(1,10).todevice();
	auto Y = Tensor<T>(M * K).rdrange(1,10).todevice();
	auto Z = Tensor<T>(N * K);
	auto Y_T = Tensor<T>(M * K, 0, 1);
	measurel("malloc & cudaMemcpy")

	float c1=0,c2=0;
	int cases = 500;
	for(int cas = 0; cas < cases; ++cas) {
		// transpose_wrapper(Y, Y2, M, K);
		// measurec("transpose kernel", c1);
		matmul<<<dim3(N / LX, K / LY), D>>>(X.d(), Y.d(), Z.d(), N, M, K);
		cudaDeviceSynchronize();
	}
	measurec("matmul kernel", c2);
	printf("transpose kernel average: %.3lfms\n", c1 / cases);
	printf("matmul kernel average: %.3lfms\n", c2 / cases);
	printf("total matmul average: %.3lfms\n", (c1 + c2) / cases);
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
