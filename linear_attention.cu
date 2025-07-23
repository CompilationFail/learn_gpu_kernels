#include "utils.cu"

constexpr int THREADS = 128;
__global__ void linear_attention_update(const T *kv_cache, const Thalf *q, const Thalf *k, const Thalf *v,
	const Thalf *slope_rate, T *new_kv_cache, T *output, int B, int H, int D) {
	int b = blockIdx.x; // batch index
	int h = blockIdx.y; // head index
	T slope = exp((T)-slope_rate[h]);
	int i0 = threadIdx.x;
	const T *kv = &kv_cache[b * H * D * D + h * D * D];
	T *new_kv = &new_kv_cache[b * H * D * D + h * D * D];
	__shared__ T vv[1024], nv[1024];
	for(int j = i0; j < D; j += THREADS) {
		vv[j] = v[b * H * D + h * D + j]; // v[b][h][j]
		// qq[j] = q[b * H * D + h * D + j]; // q[b][h][j]
		nv[j] = 0;
    }
	__syncthreads();
	for(int i = 0; i < D; ++i) {
		T ki = k[b * H * D + h * D + i]; // q[b][h][i]
		T qi = q[b * H * D + h * D + i]; // q[b][h][i]
		int base = i * D;
		for(int j = i0; j < D; j += THREADS) {
			T value = kv[base + j] * slope + ki * vv[j];
			new_kv[base + j] = value;
			nv[j] += qi * value;
		}
	}
	__syncthreads();
	T *output_base = &output[b * H * D + h * D];
	for(int j = i0; j < D; j += THREADS) 
		output_base[j] = nv[j];
}	

// inside, every ptr is cuda memory address
// returns: new_v, new_kv_cache
/*std::pair <T*, T*> linear_attention_decode(Thalf *q, Thalf *k, Thalf *v, Thalf *slope_rate, T *kv_cache, int B, int H, int D) {
	T *new_kv_cache, *output;
	cudaMalloc((void**)&new_kv_cache, sizeof(T) * B * H * D * D);
	cudaMalloc((void**)&output, sizeof(T) * B * H * D);
	dim3 blocks(B, H);
	linear_attention_update <<<blocks, THREADS>>> (kv_cache, q, k, v, slope_rate, new_kv_cache, output, B, H, D);
	return {output, new_kv_cache};
}*/

int test_linear_attention_decode(bool verify, int B, int H, int D) {
	auto last = std::chrono::high_resolution_clock::now();
	/*auto q = (Tensor<Thalf>(B * H * D).rd01().todevice()); 
	auto k = Tensor<Thalf>(B * H * D).rd01().todevice(); 
	auto v = Tensor<Thalf>(B * H * D).rd01().todevice(); 
	auto slope = Tensor<Thalf>(H).rd01().todevice(); 
	auto kv_cache = Tensor<T>(B * H * D * D).rd01().todevice(); 
	auto new_kv_cache = Tensor<T>(B * H * D * D);
	auto output = Tensor<T>(B * H * D);*/
	auto q = Tensor<Thalf>(B * H * D).rd01();
	auto k = Tensor<Thalf>(B * H * D).rd01();
	auto v = Tensor<Thalf>(B * H * D).rd01();
	auto slope = Tensor<Thalf>(H).rd01();
	auto kv_cache = Tensor<T>(B * H * D * D).rd01();
	auto new_kv_cache = Tensor<T>(B * H * D * D);
	auto output = Tensor<T>(B * H * D);
	measurel("malloc & randn")
	q.todevice();
	k.todevice();
	v.todevice();
	slope.todevice();
	kv_cache.todevice();
	measurel("copy to gpu");
	float c = 0;
	int cases = 500;
	assert(D % 128 == 0);
	for(int cas = 0; cas < cases; ++cas) {
		{
			// allow on test, comparing to torch code
			// auto new_kv_cache = Tensor<T>(B * H * D * D);
			// auto output = Tensor<T>(B * H * D);
			dim3 blocks(B, H);
			linear_attention_update <<<blocks, THREADS>>> 
				(kv_cache.d(), q.d(), k.d(), v.d(), slope.d(), new_kv_cache.d(), output.d(), B, H, D);
			cudaDeviceSynchronize();
		}
	}
	measurec("linear_attention_decode: kernel", c);
	printf("linear_attention_decode kernel average: %.3lfms\n", c / cases);
	
	checkCudaFail();
	output.tohost();
	new_kv_cache.tohost();
	checkCudaFail();
	
	int flag = 1;
	if(verify) {
		auto _k = (Thalf (*)[H][D])k.h();
		auto _q = (Thalf (*)[H][D])q.h();
		auto _v = (Thalf (*)[H][D])v.h();
		auto _slope = (Thalf (*))slope.h();
		auto _kv = (T (*)[H][D][D])kv_cache.h();
		auto _new_kv = (T (*)[H][D][D])new_kv_cache.h();
		auto _output = (T (*)[H][D])output.h();
		for(int i = 0; i < B; ++i) {
			for(int j = 0; j < H; ++j) {
				T sl = exp(-(T)_slope[j]); 
				static T buf[10000];
				for(int a = 0; a < D; ++a) buf[a]=0;
				for(int a = 0; a < D; ++a) {
					for(int b = 0; b < D; ++b) {
						T value = _kv[i][j][a][b] * sl + (T)_k[i][j][a] * (T)_v[i][j][b];
						if(!fp_equal(_new_kv[i][j][a][b], value)) {
							flag = 0;
							printf("kv:[%d][%d][%d][%d] = %.3lf != %.3lf\n", i, j, a, b, _new_kv[i][j][a][b],value);
						}
						buf[b] += (T)_q[i][j][a] * value;
					}
				}
				for(int a = 0; a < D; ++a) {
					if(!fp_equal(_output[i][j][a], buf[a])) {
						flag = 0;
						printf("output:[%d][%d][%d] = %.3lf != %.3lf\n", i, j, a, _output[i][j][a], buf[a]);
					}
				}
			}
		}
	}
	measurel("check");
	return !flag;
}
