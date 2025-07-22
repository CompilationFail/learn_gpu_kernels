#pragma once
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cassert>
#include <cuda_fp16.h>
#include <algorithm>
#include <utility>

using T = float;
using Thalf = __half;

#define measurelc(x,counter) { \
	auto now = std::chrono::high_resolution_clock::now();\
	auto count##counter = std::chrono::duration<double, std::milli>(now - last).count();\
	printf("%s: %s: %.3lfms\n", __func__, x, count##counter);\
	counter += count##counter;\
	last = now;\
} 
#define measurec(x,counter) { \
	auto now = std::chrono::high_resolution_clock::now();\
	auto count##counter = std::chrono::duration<double, std::milli>(now - last).count();\
	counter += count##counter;\
	last = now;\
} 
#define measurel(x) { \
	auto now = std::chrono::high_resolution_clock::now();\
	printf("%s: %s: %.3lfms\n", __func__, x, std::chrono::duration<double, std::milli>(now - last).count());\
	last = now;\
} 
#define checkCudaFail() {\
	cudaError_t err = cudaGetLastError();\
	if(err != cudaSuccess) {\
		printf("%s: CUDA kernel failed: %s\n", __func__, cudaGetErrorString(err));\
		return EXIT_FAILURE;\
	}\
}

template <class T> bool fp_equal(T a, T b) {
	return fabs((float)a - (float)b) < 1e-3;
}

template <class T> class Tensor {
	T *x, *d_x;
	int *counter;
	int N, size;
public:
	Tensor() = delete;
	Tensor(int _N) { 
		N = _N;
		size = sizeof(T) * N;
		x = (T*)malloc(size), d_x = nullptr;
		cudaError_t err = cudaMalloc((void**)&d_x, size);
		counter = (int*)malloc(sizeof(int));
		*counter = 1;
		if(err != cudaSuccess) {
			d_x = nullptr;
			printf("CUDA malloc fail: N=%d\n", N);
		} else { 
			// printf("Tensor malloc(%d) sizeof(T)=%d, %p %p, %p\n", N, (int)sizeof(T), x, d_x, counter);
		}
	}
	Tensor<T>(const Tensor <T> &other) {
		x = other.x, d_x = other.d_x;
		counter = other.counter, (*counter)++;
		N = other.N, size = other.size;
		// printf("copy %p %p %p counter=%d\n", x, d_x, counter, *counter);
	}
	Tensor<T>& operator=(Tensor<T>&& other)
	{
		x = other.x, d_x = other.d_x;
		counter = other.counter, counter++;
		N = other.N, size = other.size;
		return *this;
	}
	Tensor<T>& operator=(const Tensor<T>& other)
	{
		x = other.x, d_x = other.d_x;
		counter = other.counter, counter++;
		N = other.N, size = other.size;
		return *this;
	}

	~Tensor() { 
		if(--*counter == 0) {
			// printf("free: deconstruct %p, %p, %p, counter=%d\n", x, d_x, counter, *counter);
			free(x); cudaFree(d_x); 
			free(counter);
		}
	}
	Tensor& rdrange(double l, double r) {
		std::mt19937 eng(618);
		for(int i = 0; i < N; ++i) x[i] = (T)std::uniform_real_distribution<double>(l,l)(eng);
		return *this;
	}
	Tensor& rd01() { return rdrange(0,1); }
	Tensor& todevice() {
		cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
		return *this;
	} 
	Tensor& tohost() {
		cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
		return *this;
	}
	T* d() { return d_x; } // device
	T* h() { return x; }   // host
	// T& operator [] (int i) { return x[i]; }
};

