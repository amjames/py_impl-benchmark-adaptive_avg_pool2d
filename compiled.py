Output code: 

from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_ajames/td/ctdyhhxvxvxpss3nqjomz3nx4ushvku6jq3fs73l57mxj6c5fqk5.py
# Source Nodes: [], Original ATen: []

triton_per_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3096
    rnumel = 129
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 129
    r2 = rindex
    x1 = (xindex // 129)
    x3 = xindex
    tmp0 = ((256*x0) // 129)
    tmp1 = ((384 + (256*x0)) // 129)
    tmp2 = tmp0 < tmp1
    tmp3 = ((256*r2) // 129)
    tmp4 = ((384 + (256*r2)) // 129)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + ((256*((256*x0) // 129)) + (65536*x1) + ((256*r2) // 129)), rmask & tmp6 & xmask, other=0.0)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = 1 + ((256*r2) // 129)
    tmp11 = tmp10 < tmp4
    tmp12 = tmp2 & tmp11
    tmp13 = tl.load(in_ptr0 + (1 + (256*((256*x0) // 129)) + (65536*x1) + ((256*r2) // 129)), rmask & tmp12 & xmask, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp15 + tmp9
    tmp17 = 2 + ((256*r2) // 129)
    tmp18 = tmp17 < tmp4
    tmp19 = tmp2 & tmp18
    tmp20 = tl.load(in_ptr0 + (2 + (256*((256*x0) // 129)) + (65536*x1) + ((256*r2) // 129)), rmask & tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp22 + tmp16
    tmp24 = 1 + ((256*x0) // 129)
    tmp25 = tmp24 < tmp1
    tmp26 = tmp25 & tmp5
    tmp27 = tl.load(in_ptr0 + (256 + (256*((256*x0) // 129)) + (65536*x1) + ((256*r2) // 129)), rmask & tmp26 & xmask, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tmp29 + tmp23
    tmp31 = tmp25 & tmp11
    tmp32 = tl.load(in_ptr0 + (257 + (256*((256*x0) // 129)) + (65536*x1) + ((256*r2) // 129)), rmask & tmp31 & xmask, other=0.0)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tmp34 + tmp30
    tmp36 = tmp25 & tmp18
    tmp37 = tl.load(in_ptr0 + (258 + (256*((256*x0) // 129)) + (65536*x1) + ((256*r2) // 129)), rmask & tmp36 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp35
    tmp41 = 2 + ((256*x0) // 129)
    tmp42 = tmp41 < tmp1
    tmp43 = tmp42 & tmp5
    tmp44 = tl.load(in_ptr0 + (512 + (256*((256*x0) // 129)) + (65536*x1) + ((256*r2) // 129)), rmask & tmp43 & xmask, other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = tmp46 + tmp40
    tmp48 = tmp42 & tmp11
    tmp49 = tl.load(in_ptr0 + (513 + (256*((256*x0) // 129)) + (65536*x1) + ((256*r2) // 129)), rmask & tmp48 & xmask, other=0.0)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp48, tmp49, tmp50)
    tmp52 = tmp51 + tmp47
    tmp53 = tmp42 & tmp18
    tmp54 = tl.load(in_ptr0 + (514 + (256*((256*x0) // 129)) + (65536*x1) + ((256*r2) // 129)), rmask & tmp53 & xmask, other=0.0)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp53, tmp54, tmp55)
    tmp57 = tmp56 + tmp52
    tmp58 = 1.0
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp6, tmp58, tmp59)
    tmp61 = tl.where(tmp12, tmp58, tmp59)
    tmp62 = tmp61 + tmp60
    tmp63 = tl.where(tmp19, tmp58, tmp59)
    tmp64 = tmp63 + tmp62
    tmp65 = tl.where(tmp26, tmp58, tmp59)
    tmp66 = tmp65 + tmp64
    tmp67 = tl.where(tmp31, tmp58, tmp59)
    tmp68 = tmp67 + tmp66
    tmp69 = tl.where(tmp36, tmp58, tmp59)
    tmp70 = tmp69 + tmp68
    tmp71 = tl.where(tmp43, tmp58, tmp59)
    tmp72 = tmp71 + tmp70
    tmp73 = tl.where(tmp48, tmp58, tmp59)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp53, tmp58, tmp59)
    tmp76 = tmp75 + tmp74
    tmp77 = tmp57 / tmp76
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK, RBLOCK])
    tmp80 = tl.where(rmask & xmask, tmp78, 0)
    tmp81 = tl.sum(tmp80, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp81, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_ajames/qc/cqc6tp4nntwykebhaccrkibhb3r4sfzzb6u4fzaohvjlv7jtsnjk.py
# Source Nodes: [], Original ATen: []

triton_per_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 129
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (129*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ajames/6r/c6r2qtrdj574b6o27bs3jsbdpowmoqrgwzhc3xgpfd435g4fwwyh.py
# Source Nodes: [], Original ATen: []

triton_per_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_2', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ajames/gc/cgctofm6rwn3hhors6bxr552ohvilx7mof65qzkmst4qrzu35w5n.py
# Source Nodes: [], Original ATen: []

triton_per_fused_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 24
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp5 + tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    args_1, args_2 = args
    args.clear()
    assert_size_stride(args_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(args_2, (8, 3, 129, 129), (49923, 16641, 129, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((8, 3, 129), (387, 129, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_per_fused_0.run(args_1, buf3, 3096, 129, grid=grid(3096), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf1 = aten._adaptive_avg_pool2d_backward.default(args_2, args_1)
        del args_1
        del args_2
        buf2 = buf1
        del buf1
        buf4 = empty_strided_cuda((8, 3), (3, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_per_fused_1.run(buf3, buf4, 24, 129, grid=grid(24), stream=stream0)
        del buf3
        buf5 = empty_strided_cuda((8, 3, 256), (768, 256, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_per_fused_2.run(buf2, buf5, 6144, 256, grid=grid(6144), stream=stream0)
        del buf2
        buf7 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        triton_per_fused_3.run(buf7, buf5, 24, 256, grid=grid(24), stream=stream0)
        del buf5
    return (buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    args_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    args_2 = rand_strided((8, 3, 129, 129), (49923, 16641, 129, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([args_1, args_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

Output code written to: /tmp/torchinductor_ajames/u2/cu2urqot2tgtc5fib54lg4gnkvbmy3yt6ntaesmyiqjblaburafj.py
