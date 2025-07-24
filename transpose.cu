
#include "utils.cu"

// 32 * 32, so that each sub row(32 floats) fit into a single L1 transaction
// L1 Cache is write through to L2!!!!
// must provide continuity for write, 
// so that each warp can merge as a transaction to L2
// 尽量保证 warp 的一轮中, 读/写的部分是连续的, 也就是每 32 个 threads 的 warp 在同一行连续读

/*static constexpr int TransposeD = 32;
static constexpr int transpose_thread_dim_x = 4;
static constexpr int transpose_thread_dim_y = 32;
__global__ void transpose(const T *x, T *y, int N, int M) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	static constexpr int D = TransposeD;
	__shared__ T tmp[D][D];
	int ibase = blockIdx.x * D;
	int jbase = blockIdx.y * D;
	for(int i0 =  i; i0 < D; i0 += transpose_thread_dim_x) {
		for(int j0 = j; j0 < D; j0 += transpose_thread_dim_y) {
			// tmp[j0][i0] = x[(i0 + ibase) * M + j0 + jbase];
			// swiss to avoid bank conflict
			tmp[j0][(i0 + j0) % D] = x[(i0 + ibase) * M + j0 + jbase];
		}
	}
	__syncthreads();
	// i + ibase, j + jbase -> j + jbase , i + ibase
	// i + ibase, j + jbase -> j, i -> j + jbase , i + ibase
	for(int i0 =  i; i0 < D; i0 += transpose_thread_dim_x) {
		for(int j0 = j; j0 < D; j0 += transpose_thread_dim_y) {
			// y[(j0 + jbase) * N + i0 + ibase] = tmp[j0][i0];
			// swiss to avoid bank conflict
			y[(i0 + jbase) * N + j0 + ibase] = tmp[i0][(i0 + j0)%D];
		}
	}
}*/

static constexpr int TransposeD = 64;
static constexpr int transpose_thread_dim_x = 8;
static constexpr int transpose_thread_dim_y = 16;

__global__ void transpose(const T *x, T *y, int N, int M) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	static constexpr int D = TransposeD;
	__shared__ T tmp[D][D];
	int ibase = blockIdx.x * D;
	int jbase = blockIdx.y * D;
	for(int i0 =  i; i0 < D; i0 += transpose_thread_dim_x) {
		for(int j0 = j; j0 < D; j0 += transpose_thread_dim_y) {
			// tmp[j0][i0] = x[(i0 + ibase) * M + j0 + jbase];
			// swiss to avoid bank conflict
			tmp[j0][(i0 + 2*j0) % D] = x[(i0 + ibase) * M + j0 + jbase];
		}
	}
	__syncthreads();
	for(int i0 =  i; i0 < D; i0 += transpose_thread_dim_x) {
		for(int j0 = j; j0 < D; j0 += transpose_thread_dim_y) {
			// y[(j0 + jbase) * N + i0 + ibase] = tmp[j0][i0];
			// swiss to avoid bank conflict
			y[(i0 + jbase) * N + j0 + ibase] = tmp[i0][(i0*2 + j0)%D];
		}
	}
}

template <class T> void transpose_wrapper(Tensor <T> &x, Tensor <T>&y, int n, int m) {
	static constexpr int D = TransposeD;
	dim3 threads_per_block(transpose_thread_dim_x, transpose_thread_dim_y);
	dim3 blocks(n / D, m / D);
	transpose<<<blocks, threads_per_block>>>(x.d(), y.d(), n, m);
	cudaDeviceSynchronize();
}

int test_transpose(bool verify, int n, int m) {
	auto last = std::chrono::high_resolution_clock::now();
	int N = n * m;
	auto x = Tensor<T>(N).rd01().todevice();
	auto y = Tensor<T>(N);
	measurel("malloc & cudaMemcpy")

	float c=0;
	int cases = 500;
	assert(n % 128 == 0);
	assert(m % 128 == 0);
	for(int cas = 0; cas < cases; ++cas) {
		transpose_wrapper(x, y, n, m);
	}
	measurec("transpose: kernel", c);
	printf("transpose kernel average: %.3lfms\n", c / cases);
	y.tohost();
	checkCudaFail();
	int flag = 1;
	if(verify) {
		auto xx = (T (*)[m])x.h();
		auto yy = (T (*)[n])y.h();
		for(int i = 0; i < n; ++i) {
			for(int j = 0; j < m; ++j) {
				if(xx[i][j] != yy[j][i]) {
					flag = 0;
					printf("i=%d,j=%d,x=%.3lf y=%.3lf\n", i,j,xx[i][j],yy[j][i]);
				}
			}
		}
	}
	measurel("check")
	return !flag;
}
