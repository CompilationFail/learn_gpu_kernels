# ARCH for 4080 = 89, see https://developer.nvidia.com/cuda-gpus
compute_arch = 89
# warn for too many registers, enable kernel code lambda, enable kernel code constexpr, enable O3, choose arch
nvcc_cmd := nvcc -std=c++17\
	--ptxas-options=-warn-spills\
	-expt-extended-lambda\
	-expt-relaxed-constexpr\
	-Xcompiler=-O3\
	-gencode arch=compute_${compute_arch},code=sm_${compute_arch}

nsight_cmd :=nsys profile\
	--trace cuda,cudahw,osrt,nvtx,cublas\
	--cuda-memory-usage true\
	--force-overwrite true

kernel: kernel.cu softmax.cu utils.cu transpose.cu swiGLU.cu vecadd.cu linear_attention.cu matmul.cu matmul2.cu
	${nvcc_cmd} -o $@ $@.cu

profile: kernel
	${nsight_cmd} ./kernel ${testname} && nsys analyze report1.nsys-rep 

clean:
	rm kernel
