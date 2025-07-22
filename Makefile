# ARCH for 4080 = 89, see https://developer.nvidia.com/cuda-gpus
compute_arch = 89
# warn for too many registers, enable kernel code lambda, enable kernel code constexpr, enable O3, choose arch
nvcc_cmd := nvcc -std=c++17\
	--ptxas-options=-warn-spills\
	-expt-extended-lambda\
	-expt-relaxed-constexpr\
	-Xcompiler=-O3\
	-gencode arch=compute_${compute_arch},code=sm_${compute_arch}

kernel: kernel.cu softmax.cu utils.cu transpose.cu swiGLU.cu vecadd.cu linear_attention.cu
	${nvcc_cmd} -o $@ $@.cu

clean:
	rm kernel
