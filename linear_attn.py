import torch
import time
def linear_attentention_decode(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	slope_rate: torch.Tensor,
	kv_cache: torch.Tensor,
) -> torch.Tensor:
	q, k, v, slope_rate = (x.to(torch.float32) for x in (q, k, v, slope_rate))
	ratio = torch.exp(-slope_rate)
	cur_kv = torch.einsum("... s d, ... s e -> ... d e",
		k[:, :, 0:1, :], v[:, :, 0:1, :])
	print(kv_cache.shape, ratio.shape)
	new_kv_cache = kv_cache.mul_(ratio.view(1,-1,1,1)).add(cur_kv)
	output = torch.einsum("... s e, ... e d -> ... s d", q[:,:,0:1,:], new_kv_cache)
	# output = torch.rearrange(output, "b h s d -> s b (h d)")
	return output.reshape(1, B, H * D), new_kv_cache


B = 128
H = 64
D = 128

q = torch.rand(B, H, 1, D, dtype=torch.float16)
k = torch.rand(B, H, 1, D, dtype=torch.float16)
v = torch.rand(B, H, 1, D, dtype=torch.float16)
slope = torch.rand(H, dtype=torch.float16)
kv_cache = torch.rand(B, H, D, D, dtype=torch.float32)

for i in range(0,100):
	now = time.time()
	linear_attentention_decode(q, k, v, slope, kv_cache)
	print("%.3lfms" % (1000 * ( time.time()-now)))