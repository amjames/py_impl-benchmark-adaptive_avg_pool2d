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


# kernel path: /tmp/torchinductor_ajames/5s/c5sxnmwqauobzib4chclbevrwcpkfsl4kkyt4dw4kj46rkek4a3a.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3594456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3) % 129
    x0 = xindex % 3
    x3 = (xindex // 1161) % 129
    x2 = (xindex // 387) % 3
    x4 = (xindex // 149769)
    x8 = xindex
    tmp0 = 384 + (256*x1)
    tmp1 = tl.full([1], 129, tl.int64)
    tmp2 = tmp0 // tmp1
    tmp3 = 256*x1
    tmp4 = tmp3 // tmp1
    tmp5 = tmp2 - tmp4
    tmp6 = x0
    tmp7 = tmp6 >= tmp5
    tmp8 = 384 + (256*x3)
    tmp9 = tmp8 // tmp1
    tmp10 = 256*x3
    tmp11 = tmp10 // tmp1
    tmp12 = tmp9 - tmp11
    tmp13 = x2
    tmp14 = tmp13 >= tmp12
    tmp15 = tmp11 + tmp13
    tmp16 = tl.full([1], 255, tl.int64)
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp4 + tmp6
    tmp19 = triton_helpers.minimum(tmp18, tmp16)
    tmp20 = tl.load(in_ptr0 + (tmp19 + (256*tmp17) + (65536*x4)), xmask, eviction_policy='evict_last')
    tmp21 = 0.0
    tmp22 = tl.where(tmp14, tmp21, tmp20)
    tmp23 = tl.where(tmp7, tmp21, tmp22)
    tl.store(out_ptr0 + (x8), tmp23, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_ajames/za/czax532md7mhwiwad3rgzqipuxzdflxkyr73pqwpjjiqpsgds3cs.py
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
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'}
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
    r1 = rindex
    x0 = xindex
    x2 = xindex % 129
    tmp0 = tl.load(in_ptr0 + ((3*r1) + (1161*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr0 + (1 + (3*r1) + (1161*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr0 + (2 + (3*r1) + (1161*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr0 + (387 + (3*r1) + (1161*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr0 + (388 + (3*r1) + (1161*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr0 + (389 + (3*r1) + (1161*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr0 + (774 + (3*r1) + (1161*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr0 + (775 + (3*r1) + (1161*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr0 + (776 + (3*r1) + (1161*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 384 + (256*x2)
    tmp18 = tl.full([1, 1], 129, tl.int64)
    tmp19 = tmp17 // tmp18
    tmp20 = 256*x2
    tmp21 = tmp20 // tmp18
    tmp22 = tmp19 - tmp21
    tmp23 = 384 + (256*r1)
    tmp24 = tmp23 // tmp18
    tmp25 = 256*r1
    tmp26 = tmp25 // tmp18
    tmp27 = tmp24 - tmp26
    tmp28 = tmp22 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp16 / tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp34, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ajames/fk/cfkwmpfi26fr3zfxofrdbmzhqr6sb7khnt24i3i6dszniemm2xmz.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ajames/qt/cqtlwputhm4o3vuqjewpktnagsf6i5ea25sm3hxwv3zvbwzdazb7.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': ['out_ptr8'], 'no_x_dim': False, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3594456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 387) % 3
    x0 = xindex % 3
    x1 = (xindex // 3) % 129
    x5 = (xindex // 1161)
    x3 = (xindex // 1161) % 129
    x8 = xindex
    x4 = (xindex // 149769)
    tmp5 = tl.load(in_ptr0 + (x1 + (129*x5)), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tmp3 == tmp1
    tmp6 = 384 + (256*x3)
    tmp7 = tl.full([1], 129, tl.int64)
    tmp8 = tmp6 // tmp7
    tmp9 = 256*x3
    tmp10 = tmp9 // tmp7
    tmp11 = tmp8 - tmp10
    tmp12 = 384 + (256*x1)
    tmp13 = tmp12 // tmp7
    tmp14 = 256*x1
    tmp15 = tmp14 // tmp7
    tmp16 = tmp13 - tmp15
    tmp17 = tmp11 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp5 / tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tl.where(tmp2, tmp21, tmp20)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp3 == tmp23
    tmp25 = tl.where(tmp24, tmp19, tmp20)
    tmp26 = tl.where(tmp2, tmp25, tmp20)
    tmp27 = tl.full([1], 0, tl.int32)
    tmp28 = tmp3 == tmp27
    tmp29 = tl.where(tmp28, tmp19, tmp20)
    tmp30 = tl.where(tmp2, tmp29, tmp20)
    tmp31 = tmp0 == tmp23
    tmp32 = tl.where(tmp31, tmp21, tmp20)
    tmp33 = tl.where(tmp31, tmp25, tmp20)
    tmp34 = tl.where(tmp31, tmp29, tmp20)
    tmp35 = tmp0 == tmp27
    tmp36 = tl.where(tmp35, tmp21, tmp20)
    tmp37 = tl.where(tmp35, tmp25, tmp20)
    tmp38 = tl.where(tmp35, tmp29, tmp20)
    tmp39 = tmp22 + tmp26
    tmp40 = tmp39 + tmp30
    tmp41 = tmp40 + tmp32
    tmp42 = tmp41 + tmp33
    tmp43 = tmp42 + tmp34
    tmp44 = tmp43 + tmp36
    tmp45 = tmp44 + tmp37
    tmp46 = tmp45 + tmp38
    tmp47 = tmp10 + tmp0
    tmp48 = tl.full([1], 255, tl.int64)
    tmp49 = triton_helpers.minimum(tmp47, tmp48)
    tmp50 = tmp15 + tmp3
    tmp51 = triton_helpers.minimum(tmp50, tmp48)
    tmp52 = tmp0 >= tmp11
    tmp53 = tmp3 >= tmp16
    tmp54 = tl.where(tmp53, tmp20, tmp46)
    tmp55 = tl.where(tmp52, tmp20, tmp54)
    tl.atomic_add(out_ptr8 + (tmp51 + (256*tmp49) + (65536*x4)), tmp55, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_ajames/wv/cwvqupndcpg5zxzzvbhy6h74xg6cvkfop7c657zxce5tvtsp3vyv.py
# Source Nodes: [], Original ATen: []

triton_per_fused_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'}
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


# kernel path: /tmp/torchinductor_ajames/r4/cr4tihqfhbr2m2xodd26qjvjktpm7jvvw6njrbcm4gsf7fy6c5fd.py
# Source Nodes: [], Original ATen: []

triton_per_fused_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_5', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'}
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


# kernel path: /tmp/torchinductor_ajames/gc/cgcepp7btuedfd7lt2pwhe7jhozbq252hjzgzxwqsi2l7csjjdzb.py
# Source Nodes: [], Original ATen: []

triton_per_fused_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'backend_hash': '22f22a3619290574d4d20d0a2299c2823f933d677b41781e9e1e30c4f1ac3355'}
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
        buf0 = empty_strided_cuda((8, 3, 129, 3, 129, 3), (449307, 149769, 1161, 387, 3, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(args_1, buf0, 3594456, grid=grid(3594456), stream=stream0)
        del args_1
        buf14 = empty_strided_cuda((8, 3, 129), (387, 129, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_per_fused_1.run(buf0, buf14, 3096, 129, grid=grid(3096), stream=stream0)
        del buf0
        buf12 = empty_strided_cuda((8, 3, 256, 256), (196608, 65536, 256, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf12, 1572864, grid=grid(1572864), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(args_2, buf12, 3594456, grid=grid(3594456), stream=stream0)
        del args_2
        buf15 = empty_strided_cuda((8, 3), (3, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_per_fused_4.run(buf14, buf15, 24, 129, grid=grid(24), stream=stream0)
        del buf14
        buf16 = empty_strided_cuda((8, 3, 256), (768, 256, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_per_fused_5.run(buf12, buf16, 6144, 256, grid=grid(6144), stream=stream0)
        del buf12
        buf18 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: []
        triton_per_fused_6.run(buf18, buf16, 24, 256, grid=grid(24), stream=stream0)
        del buf16
    return (buf18, )


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

Output code written to: /tmp/torchinductor_ajames/m4/cm4eeqrco5dyikpxl6z3tagthjxefipktfmm6ukcjaism5kdpyb2.py
