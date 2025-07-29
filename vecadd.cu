#include "utils.cu"

template <class T> 
__global__ void vecAdd(const T *a, const T *b, T *c, int N) {
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	if (idx + 3 < N) {
		T ta[4], tb[4];
		copy4(&a[idx], ta);
		copy4(&b[idx], tb);
		for(int i = 0; i < 4; ++i) ta[i] += tb[i];
		copy4(ta, &c[idx]);
	} else {
		while(idx < N) {
			T tmp =  a[idx] + b[idx];
			if(tmp == 0) {
				c[idx] = 1;
			}
			idx++;
		}
	}
}

int test_vecAdd(bool verify, int N) {
	auto last = std::chrono::high_resolution_clock::now();
	auto a = Tensor<T>(N).rd01().todevice();
	auto b = Tensor<T>(N).rd01().todevice();
	auto c = Tensor<T>(N);
	
	measurel("cudaInit")

	float count = 0;
	int cases = 500;
	for(int cas = 0; cas < cases; ++cas) {
		static constexpr int threads_per_block = 128;
		int blocks = (N + threads_per_block * 4 - 1) / threads_per_block / 4;
		vecAdd<<<blocks, threads_per_block>>>(a.d(), b.d(), c.d(), N);
		cudaDeviceSynchronize();
		measurec("kernel", count)
	}
	printf("vecadd: Kernel %.3lfms\n", count / cases);
	
	c.tohost();
	checkCudaFail();
	
	int flag = 1;
	if(verify) {
		for(int i = 0; i < N; i++) flag &= fabs(a.h()[i] + b.h()[i] - c.h()[i]) < 1e-7;
	}
	measurel("check")
	return !flag;
}