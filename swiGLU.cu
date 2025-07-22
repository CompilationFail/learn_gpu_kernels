#include "utils.cu"

template <class T> 
__global__ void SwiGLU(const T *x_proj, T *y, int N, int hidden_dim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		int i = idx % hidden_dim;
		int ix1 = (idx - i) * 2 + i, ix2 = ix1 + hidden_dim;
		T x1 = x_proj[ix1];
		T x2 = x_proj[ix2];
		y[idx] = x1 / (1 + exp(-x1)) * x2;
	}
}

template <class T> 
__global__ void SwiGLU2(const T *x_proj, T *y, int N, int hidden_dim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_threads = gridDim.x * blockDim.x;
	for(; idx < N; idx += total_threads) {
		int i = idx % hidden_dim;
		int ix1 = (idx - i) * 2 + i, ix2 = ix1 + hidden_dim;
		T x1 = x_proj[ix1];
		T x2 = x_proj[ix2];
		y[idx] = x1 / (1 + exp(-x1)) * x2;
	}
}

int test_swiGLU(bool verify, int batch, int seq_len, int hidden_dim) {
	auto last = std::chrono::high_resolution_clock::now();
	int N  = batch * seq_len * hidden_dim;
	auto x_proj = Tensor<T>(N * 2).rdrange(-1,1).todevice();
	auto y = Tensor<T>(N);
	measurel("init")

	float c1 = 0;
	int cases = 500;
	for(int cas = 0; cas < cases; ++cas) {
		{
			static constexpr int threads_per_block = 128;
			int blocks = (N + threads_per_block - 1) / threads_per_block;
			SwiGLU<<<blocks, threads_per_block>>>(x_proj.d(), y.d(), N, hidden_dim);
			cudaDeviceSynchronize();
		}
	}
	measurec("swiGLU: kernel", c1);
	printf("swiGLU1 average: %.3lfms\n", c1 / cases);
	/*float c2 = 0;
	for(int cas = 0; cas < cases; ++cas) {
		{
			SwiGLU2<<<16384, 128>>>(d_x_proj, d_y, N, hidden_dim);
			cudaDeviceSynchronize();
			measurec("swiGLU2: kernel", c2);
		}
	}
	printf("swiGLU2 average: %.3lfms\n", c2 / cases);*/
	y.tohost();
	checkCudaFail();
	int flag = 1;
	if(verify) {
		T (*x)[seq_len][hidden_dim * 2] = (T (*)[seq_len][hidden_dim * 2])x_proj.h();
		T (*yy)[seq_len][hidden_dim] = (T (*)[seq_len][hidden_dim])y.h();
		for(int i = 0; i < batch; ++i) {
			for(int j = 0; j < seq_len; ++j) {
				for(int k = 0; k < hidden_dim; ++k) {
					T x1 = x[i][j][k];
					T x2 = x[i][j][k+hidden_dim];
					T swiGLU = x1 / (1 + exp(-x1)) * x2;
					if(fabs(yy[i][j][k]-swiGLU) > 1e-3) {
						flag = 0;
						printf("i=%d,j=%d,k=%d,x1=%.3lf x2=%.3lf swiGLU=%.7lf y=%.7lf\n", i,j,k,x1, x2, swiGLU, yy[i][j][k]);
					}
				}
			}
		}
	}
	measurel("swiGLU: clean & leave")
	return !flag;
}