import numpy
import torch
from torch.nn.parameter import Parameter
import numpy as np

try:
    import autogptq_gptq_marlin_cuda

    major, minor = torch.cuda.get_device_capability()
    device_capability = major * 10 + minor
    if device_capability < 80:
        raise 'TODO'
except ImportError as e:
    gptq_marlin_import_exception = e

    def error_raiser_gptq_marlin(*args, **kwargs):
        raise ValueError(
            f"Trying to use the gptq marlin backend, but could not import the C++/CUDA dependencies with the following error: {marlin_import_exception}"
        )

    autogptq_gptq_marlin_cuda = gptq_marlin_import_exception

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

GPTQ_MARLIN_SUPPORTED_SYM = [True]

def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()

def get_pack_factor(num_bits):
    assert num_bits in GPTQ_MARLIN_SUPPORTED_NUM_BITS, (f"Unsupported num_bits = {num_bits}")
    return 32 // num_bits


def marlin_permute_scales(s, size_k, size_n, group_size):
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    else:
        s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()
    return s

class QuantLinear(nn.Module):
    QUANT_TYPE = "gptq_marlin"
    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        trainable=False,
        weight_dtype=torch.float16,
    ):
        self.pack_factor = get_pack_factor(bits)
        self.tile_size = GPTQ_MARLIN_TILE
        self.min_thread_n = GPTQ_MARLIN_MIN_THREAD_N
        self.min_thread_k = GPTQ_MARLIN_MIN_THREAD_K
        self.max_parallel = GPTQ_MARLIN_MAX_PARALLEL
        
        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")
        if self.group_size not in [-1, 32, 64, 128]:
            raise NotImplementedError("Only -1, 32, 64, 128 group_size are supported.")
        if weight_dtype != torch.float16:
            raise ValueError(f"The params dtype must be float16, but got {weight_dtype}")
        if infeatures % self.min_thread_k != 0:
            raise ValueError(f"infeatures = {outfeatures} is not divisible by min_thread_k = {self.min_thread_k}.")
        if outfeatures % self.min_thread_n != 0:
            raise ValueError(f"outfeatures = {outfeatures} is not divisible by min_thread_n = {self.min_thread_n}.")
        if (infeatures % group_size != 0):
            raise ValueError(f"infeatures = {infeatures} is not divisible by group_size = {group_size}.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.group_size = group_size if group_size != -1 else infeatures
        scales_and_zp_size = infeatures // group_size 
        
        self.register_buffer("qweight", torch.empty(infeatures // self.quant_config.pack_factor, outfeatures, dtype=torch.int32))
        self.register_buffer("scales", torch.empty(scales_and_zp_size, output_size_per_partition, dtype=weight_dtype))
        self.register_buffer("g_idx", torch.empty(infeatures, dtype=torch.int32))
        self.register_buffer("g_idx_sort_indices", torch.empty(g_idx.shape, dtype=torch.int32))
        self.register_buffer("workspace", torch.zeros((outfeatures // self.min_thread_n) * self.max_parallel, dtype=torch.int))
        self.scales = marlin_permute_scales(self.scales, self.infeatures, self.outfeatures, self.group_size)
        
    def pack(self, linear, scales, desc_act: bool, is_sym = False):
        if self.is_sym == True:
            raise 
            
        if desc_act and group_size == -1:
            desc_act = False

        if self.quant_config.desc_act:
            self.g_idx = self.g_idx[g_idx_sort_indices]
            self.g_idx_sort_indices = torch.argsort(self.g_idx).to(torch.int)
        else:
            self.g_idx = torch.empty(0,dtype=torch.int,device=self.qweight.device)
            self.g_idx_sort_indices = torch.empty(0, dtype=torch.int, device=self.qweight.device)
        self.qweight = ops.gptq_marlin_repack(
            self.qweight,
            self.g_idx_sort_indices,
            self.infeatures,
            self.outfeatures,
        )
    
    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        
        x = x.reshape(-1, x.shape[-1])
        output = gptq_marlin_gemm(
            x, 
            self.qweight,
            self.scales,
            self.g_idx, 
            self.g_idx_sort_indices,
            self.workspace, 
            x.shape[0], 
            self.outfeatures,
            self.infeatures, 
            True # is_k_full
        )
        
        if self.bias is not None:
            output += self.bias
        return output.reshape(out_shape)