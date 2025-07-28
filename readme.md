# GPU Kernels

Some kernel codes, tested on RTX4080 for learning purpose.

For different nvidia cards, change `compute_arch` in Makefile

usage

```bash
./nv.sh # install nvidia tool kit on linux
make kernel
./kernel [kernels] # test kernels
```

Some test results (test each kernel for 10 times; each test will run kernel for 500 times)

Expected result: Time should be almost proportional to memory access data size.

```plain
test_vecAdd for N=134217728
(Memory access: 2N floats read, N floats write)
test_vecAdd: cudaInit: 3013.725ms
test_vecAdd: check: 377.756ms
vecadd: Kernel 2.714ms
vecadd: Kernel 2.721ms
vecadd: Kernel 2.718ms
vecadd: Kernel 2.718ms
vecadd: Kernel 2.721ms
vecadd: Kernel 2.804ms
vecadd: Kernel 2.736ms
vecadd: Kernel 2.725ms
vecadd: Kernel 2.721ms
vecadd: Kernel 2.734ms


test_swiGLU for batch=128 seq_len=1024, hidden_dim=1024, N=134217728
(Memory access: 2N floats read, N floats write)
test_swiGLU: init: 2913.861ms
test_swiGLU: swiGLU: clean & leave: 639.029ms
swiGLU1 average: 2.716ms
swiGLU1 average: 2.712ms
swiGLU1 average: 2.716ms
swiGLU1 average: 2.718ms
swiGLU1 average: 2.715ms
swiGLU1 average: 2.746ms
swiGLU1 average: 2.745ms
swiGLU1 average: 2.743ms
swiGLU1 average: 2.742ms
swiGLU1 average: 2.748ms


test_softmax for batch=128 seq_len=1024, D=1024, N=134217728
(Memory access: N floats read, N floats write)
test_softmax: init: 1593.703ms
test_softmax: clean & leave: 1050.010ms
kernel average: 1.842ms
kernel average: 1.830ms
kernel average: 1.812ms
kernel average: 1.813ms
kernel average: 1.830ms
kernel average: 1.824ms
kernel average: 1.817ms
kernel average: 1.823ms
kernel average: 1.825ms
kernel average: 1.812ms

test_transpose for N=16384,M=8192,N*M=134217728
(Memory access: N*M floats read, N*M floats write)
test_transpose: malloc & cudaMemcpy: 1716.881ms
test_transpose: check: 2038.605ms
transpose kernel average: 1.821ms
transpose kernel average: 1.819ms
transpose kernel average: 1.816ms
transpose kernel average: 1.830ms
transpose kernel average: 1.831ms
transpose kernel average: 1.841ms
transpose kernel average: 1.826ms
transpose kernel average: 1.824ms
transpose kernel average: 1.832ms
transpose kernel average: 1.836ms

test_linear_attention_decode success for B*H*D=1048576, B*H*D*D=134217728
(Memory access: (BHDD floats + (3BHD+H) fp16) read, (BHD+BHDD) floats write)
test_linear_attention_decode: malloc & randn: 1520.843ms
test_linear_attention_decode: copy to gpu: 48.923ms
test_linear_attention_decode: check: 607.154ms
linear_attention_decode kernel average: 1.896ms
linear_attention_decode kernel average: 1.822ms
linear_attention_decode kernel average: 1.816ms
linear_attention_decode kernel average: 1.842ms
linear_attention_decode kernel average: 1.838ms
linear_attention_decode kernel average: 1.833ms
linear_attention_decode kernel average: 1.817ms
linear_attention_decode kernel average: 1.877ms
linear_attention_decode kernel average: 1.828ms
linear_attention_decode kernel average: 1.832ms

test_matmul success for N=512,M=512,K=64,N*M*K=16777216
test_matmul: malloc & cudaMemcpy: 159.044ms
matmul kernel average: 0.053ms
matmul kernel average: 0.047ms
matmul kernel average: 0.047ms
matmul kernel average: 0.048ms
matmul kernel average: 0.049ms
matmul kernel average: 0.052ms
matmul kernel average: 0.039ms
matmul kernel average: 0.043ms
matmul kernel average: 0.044ms
matmul kernel average: 0.044ms
```
