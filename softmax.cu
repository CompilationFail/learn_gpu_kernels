#include "utils.cu"

template <class T> 
__global__ void softmax(const T *X, T *Y, int D) {
	// now operating on arr[0~L)
	__shared__ T buf[1024];
	const T *x = &X[D * blockIdx.x];
	T *y = &Y[D * blockIdx.x];
	int idx = threadIdx.x;
	// number of threads, should be a power of 2
	int N = blockDim.x;
	
	// block reduce for max(x[0..D])
	buf[idx] = x[idx];
	for(int i = idx; (i += N) < D;) 
		buf[idx] = max(buf[idx], x[i]);
	__syncthreads();
	for(int stride = N; stride /= 2;) {
		if(idx < stride) {
			buf[idx] = max(buf[idx], buf[idx + stride]);
		}
		__syncthreads();
	}
	T ma = buf[0];
	__syncthreads();

	// block reduce for sum(exp(x[i]-ma))
	buf[idx] = 0;
	for(int i = idx; i < D; i += N)
		buf[idx] += exp(x[i] - ma);
	__syncthreads();
	for(int stride = N; stride /= 2;) {
		if(idx < stride) {
			buf[idx] += buf[idx + stride];
		}
		__syncthreads();
	}
	T sum = buf[0];

	for(int i = idx; i < D; i += N)
		y[i] = exp(x[i] - ma) / sum;
}

int test_softmax(bool verify, int batch, int seq_len, int D) {
	auto last = std::chrono::high_resolution_clock::now();
	int N  = batch * seq_len * D;
	auto x = Tensor<T>(N).rd01().todevice();
	auto y = Tensor<T>(N);
	measurel("init");

	float c = 0;
	int cases = 500;
	assert(D % 128 == 0);
	for(int cas = 0; cas < cases; ++cas) {
		{
			static constexpr int threads_per_block = 128;
			softmax<<<seq_len * batch, threads_per_block>>>(x.d(), y.d(), D);
			cudaDeviceSynchronize();
		}
	}
	measurec("softmax: kernel", c);
	printf("kernel average: %.3lfms\n", c / cases);
	
	y.tohost();
	checkCudaFail();
	int flag = 1;
	if(verify) {
		auto xx = (T (*)[seq_len][D])x.h();
		auto yy = (T (*)[seq_len][D])y.h();
		for(int i = 0; i < batch; ++i) {
			for(int j = 0; j < seq_len; ++j) {
				T ma = 0, sum = 0;
				for(int k = 0; k < D; ++k) {
					ma = max(ma, xx[i][j][k]);
				}
				for(int k = 0; k < D; ++k) {
					sum += exp(xx[i][j][k] - ma);
				}
				for(int k = 0; k < D; ++k) {
					T tmp = exp(xx[i][j][k] - ma) / sum;
					if(fabs(tmp - yy[i][j][k]) > 1e-3) {
						flag = 0;
						printf("i=%d,j=%d,k=%d, x=%.7lf, sum=%.7lf,softmax=%.7lf y=%.7lf\n", i,j,k, xx[i][j][k], sum, tmp, yy[i][j][k]);
					}
				}
			}
		}
	}
	measurel("clean & leave")
	return !flag;
}