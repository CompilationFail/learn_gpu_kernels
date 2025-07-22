#include "utils.cu"

int main(int argc, char**argv) {
	// result should be almost propotional to datasize/bandwidth
	if(argc < 2) {
		puts("usage kernel_name");
		return 0;
	}
	for(int i = 1; i < argc; ++i) {
		if(!strcmp(argv[i], "vecAdd")) {
			int N = 1 << 27;
			int rtcode = test_vecAdd(true, N);
			if(rtcode) return rtcode;
			else printf("test_vecAdd success for N=%d\n", N);
		}
		if(!strcmp(argv[i], "swiGLU")) {
			int batch = 128, seq_len = 1024, hidden_dim = 1024;
			// int batch = 4, seq_len = 4, hidden_dim = 256;
			int rtcode = test_swiGLU(true, batch, seq_len, hidden_dim);
			if(rtcode) return rtcode;
			else printf("test_swiGLU success for batch=%d seq_len=%d, hidden_dim=%d, N=%d\n", batch, seq_len, hidden_dim, batch * seq_len * hidden_dim);
		}
		if(!strcmp(argv[i], "softmax")) {
			int batch = 128, seq_len = 1024, D = 1024;
			
			// 比较下面两组会发现如果 D 容量超过 L1 Cache 就会出现倍数下降
			// int batch = 32, seq_len = 16, D = 128 * 1024;
			// int batch = 8, seq_len = 16, D = 128 * 1024 * 4;
			// int batch = 16, seq_len = 32, D = 4096 * 8 * 8;
			int rtcode = test_softmax(true, batch, seq_len, D);
			if(rtcode) return rtcode;
			else printf("test_softmax success for batch=%d seq_len=%d, D=%d, N=%d\n", batch, seq_len, D, batch * seq_len * D);
		}
		if(!strcmp(argv[i], "transpose")) {
			int N = 8192 * 2, M = 8192;
			// int N = 128, M = 128;
			int rtcode = test_transpose(true, N, M);
			if(rtcode) return rtcode;
			else printf("test_transpose success for N=%d,M=%d,N*M=%d\n", N, M, N*M);
		}
		if(!strcmp(argv[i], "linear_attention_decode")) {
			int B = 128, H = 64, D = 128;
			int rtcode = test_linear_attention_decode(true, B, H, D);
			if(rtcode) return rtcode;
			else printf("test_linear_attention_decode success for B*H*D=%d, B*H*D*D=%d\n",  B * H * D, B * H * D * D);
		}
	}
}
