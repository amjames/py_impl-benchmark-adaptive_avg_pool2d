class fn(torch.nn.Module):
    def forward(self, args):
        args_1: "f32[8, 3, 256, 256]"; args_2: "f32[8, 3, 129, 129]"; 
    
        args_1, args_2, = fx_pytree.tree_flatten_spec([args], self._in_spec)
        # No stacktrace found for following nodes
        _adaptive_avg_pool2d: "f32[8, 3, 129, 129]" = torch.ops.aten._adaptive_avg_pool2d.default(args_1, [129, 129])
        _adaptive_avg_pool2d_backward: "f32[8, 3, 256, 256]" = torch.ops.aten._adaptive_avg_pool2d_backward.default(args_2, args_1);  args_2 = args_1 = None
        alias: "f32[8, 3, 256, 256]" = torch.ops.aten.alias.default(_adaptive_avg_pool2d_backward);  _adaptive_avg_pool2d_backward = None
        alias_1: "f32[8, 3, 256, 256]" = torch.ops.aten.alias.default(alias);  alias = None
        sum_1: "f32[8, 3, 129]" = torch.ops.aten.sum.dim_IntList(_adaptive_avg_pool2d, [-1]);  _adaptive_avg_pool2d = None
        sum_2: "f32[8, 3]" = torch.ops.aten.sum.dim_IntList(sum_1, [-1]);  sum_1 = None
        sum_3: "f32[8, 3, 256]" = torch.ops.aten.sum.dim_IntList(alias_1, [-1]);  alias_1 = None
        sum_4: "f32[8, 3]" = torch.ops.aten.sum.dim_IntList(sum_3, [-1]);  sum_3 = None
        add: "f32[8, 3]" = torch.ops.aten.add.Tensor(sum_2, sum_4);  sum_2 = sum_4 = None
        return pytree.tree_unflatten([add], self._out_spec)
        