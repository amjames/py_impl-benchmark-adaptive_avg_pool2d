import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import compile_fx_inner, compile_fx
from torch._inductor.decomposition import decompositions
from torch._inductor import config
from functools import partial
from unittest.mock import patch
from torch._dispatch.python import enable_python_dispatcher
import sys


torch._logging.set_logs(output_code=True)





def run_and_get_cpp_code(fn, filename, args):
    # We use the patch context manager instead of using it as a decorator.
    # In this way, we can ensure that the attribute is patched and unpatched correctly
    # even if this run_and_get_cpp_code function is called multiple times.
    with patch.object(config, "debug", True):
        torch._dynamo.reset()
        import io
        import logging

        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        from torch._inductor.graph import output_code_log

        output_code_log.addHandler(ch)
        prev_level = output_code_log.level
        output_code_log.setLevel(logging.DEBUG)
        compiled = compile_fx_inner(fn, args, cudagraphs=True)
        result = compiled(args)
        s = log_capture_string.getvalue()
        output_code_log.setLevel(prev_level)
        output_code_log.removeHandler(ch)

    with open(filename, 'w') as f:
        f.write(s)
    return result

def make_fx_and_dump(fn, filename, args):
    fx = make_fx(fn, decomposition_table=decompositions, tracing_mode='fake')(args)
    with open(filename, 'w') as f:
        f.write(fx.print_readable())
    return fx


def fn(args):
    x, phony_grad  = args
    # passing the size this way won't cause cudagraphs to bail
    output_size = phony_grad.shape[-2:]
    out = torch.ops.aten._adaptive_avg_pool2d(x, output_size) 
    out.backward(phony_grad, retain_graph=True)
    return out.sum(-1).sum(-1) + x.grad.sum(-1).sum(-1)





arg_device = "cuda"
make_arg = partial(torch.randn, dtype=torch.float32, device=arg_device)
make_grad = partial(torch.ones, dtype=torch.float32, device=arg_device)

x = make_arg(8, 3, 256, 256, requires_grad=True)
out_sz = 129, 129
phony_grad = make_grad(8, 3, *out_sz) 
fx = make_fx_and_dump(fn, 'fx.py', [x, phony_grad]) 
result = run_and_get_cpp_code(fx, 'compiled.py', [x, phony_grad])

with enable_python_dispatcher():
    py_fx = make_fx_and_dump(fn, 'py_fx.py', [x, phony_grad])
    py_result = run_and_get_cpp_code(py_fx, 'py_compiled.py', [x, phony_grad])

