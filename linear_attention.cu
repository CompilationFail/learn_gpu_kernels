#include "utils.cu"

constexpr int THREADS = 128;
__global__ void linear_attention_update(const T *kv_cache, const Thalf *q, const Thalf *k, const Thalf *v,
	const Thalf *slope_rate, T *new_kv_cache, T *output, int B, int H, int D) {
	const int b = blockIdx.x; // batch index
	const int h = blockIdx.y; // head index
	const int i0 = threadIdx.x;
	const T slope = exp((T)-slope_rate[h]);
	const int base1 = b * H * D * D + h * D * D;
	const int base2 = b * H * D + h * D;
	const T *kv = kv_cache + base1;
	T *new_kv = new_kv_cache + base1;
	const Thalf *vbase = v + base2;
	const Thalf *kbase = k + base2;
	const Thalf *qbase = q + base2;
	T *output_base = output + base2;
	__shared__ T vv[1024], nv[1024], kk[1024], qq[1024];
	for(int j = i0; j < D; j += THREADS) {
		vv[j] = vbase[j]; // v[b][h][j]
		kk[j] = kbase[j]; 
		// 先 load 到 shared memory 会比下面 ki = kbase[i] 的时候 load 快很多
		// 解析：
		//  从 VRAM load 到 l1 的 时间一样
		//  如果下面一个个 load，有 D 次 l1 load 时间，所以会略慢
		//  但这里 l1 load 上来的时间会并行掉 (D/THREADS 次)，最后走 sharedmem 就是一个 clocktick 的时间 (读同一个位置不会 bank conflict)
		qq[j] = qbase[j];
		// qq[j] = q[b * H * D + h * D + j]; // q[b][h][j]
		nv[j] = 0;
    }
	__syncthreads();
	for(int i = 0; i < D; ++i, new_kv += D, kv += D) {
		T ki = kk[i]; // q[b][h][i]
		T qi = qq[i]; // q[b][h][i]
		for(int j = i0; j < D; j += THREADS) {
			T value = kv[j] * slope + ki * vv[j];
			new_kv[j] = value;
			nv[j] += qi * value;
		}
	}
	__syncthreads();
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
