import sys
from functools import partial
from itertools import product
from typing import Optional, Union

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._inductor.compile_fx import compile_fx_inner, cudagraphify_impl
from torch._inductor.decomposition import decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.benchmark import Compare, Timer
import logging

device = "cuda:1"
torch._logging.set_logs(all=logging.INFO, output_code=True, graph=True, graph_code=True, aot_joint_graph=True, graph_breaks=True)


def benchmark(name, label, sub_label, f, args):
    """Update signature and sub label as needed"""
    input, phony_grad = args
    assert input.requires_grad_
    return Timer(
        "f([input, phony_grad])",
        globals=locals(),
        label=name,
        description=label,
        sub_label=sub_label,
        num_threads=torch.get_num_threads(),
    ).blocked_autorange(min_run_time=2)


def gen_inputs():
    """Modify this to generate the correct args for function"""
    make_arg = partial(torch.randn, dtype=torch.float32, device=device)
    N = [8, 16, 32]
    C = [3]
    IN_SZ = [256, 512, 1024]
    OUT_FACTOR = [2, 3, 4, 5]
    for n, c, insz, out_factor in product(N, C, IN_SZ, OUT_FACTOR):
        outsz = int(insz / out_factor)
        if insz % outsz == 0:
            outsz = outsz+1
        yield (make_arg(n, c, insz, insz), make_arg(n, c, outsz, outsz))


def compile_subject(f, args):
    decomposed_py = make_fx(f, decomposition_table=decompositions, tracing_mode="fake")(args)
    return compile_fx_inner(decomposed_py, args, cudagraphs=False)


def gen_compare(name, input, phony_grad):
    """Fix signature as needed"""

    def f(args):
        """Unpack args as needed, update val=line to call correct function"""
        input, phony_grad = args
        out_sz = phony_grad.shape[-1]
        out = torch.ops.aten.adaptive_avg_pool2d(input, [out_sz, out_sz])
        out.backward(phony_grad, retain_graph=True)
        return input.grad.sum(dim=-1).sum(dim=-1) + out.sum(dim=-1).sum(dim=-1)

    out_size = phony_grad.shape[-1]
    sub_label = f"{input.shape=}, {out_size=}"
    sys.stderr.write(f"{sub_label}\n")
    input = input.clone().detach().requires_grad_(True)
    args = [input, phony_grad]
    with enable_python_dispatcher():
        py_compiled_func = compile_subject(f, [input, phony_grad])
        yield benchmark(
            name,
            "Compile-pyimpl",
            sub_label,
            py_compiled_func,
            [input, phony_grad],
        )

    compiled_func = compile_subject(f, [input, phony_grad])
    yield benchmark(
        name, "Compile", sub_label, compiled_func, [input, phony_grad]
    )
    # Just show the first two generated kernels
    torch._logging.set_logs(output_code=False)
    # Fails: element 0 of tensors does not require grad and does not have a grad_fn.
    # During the warmup execution, some of the setup drops the requires_grad_ property on the input tensor
    # eager_func = cudagraphify_impl(f, args, list(range(len(args))))
    eager_func = f
    yield benchmark(name, "Eager", sub_label, eager_func, args)


results = []
name = f"adaptive_avg_pool2d"
for args in gen_inputs():
    for res in gen_compare(name, *args):
        results.append(res)


compare = Compare(results)
compare.trim_significant_figures()
compare.print()
