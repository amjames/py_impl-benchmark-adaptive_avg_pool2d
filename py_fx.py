class fn(torch.nn.Module):
    def forward(self, args):
        args_1: "f32[8, 3, 256, 256]"; args_2: "f32[8, 3, 129, 129]"; 
    
        args_1, args_2, = fx_pytree.tree_flatten_spec([args], self._in_spec)
        # No stacktrace found for following nodes
        iota: "i64[129]" = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul: "i64[129]" = torch.ops.aten.mul.Tensor(iota, 256)
        div: "i64[129]" = torch.ops.aten.div.Tensor_mode(mul, 129, rounding_mode = 'trunc');  mul = None
        iota_1: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[129, 1]" = torch.ops.aten.unsqueeze.default(div, -1)
        add: "i64[129, 3]" = torch.ops.aten.add.Tensor(unsqueeze, iota_1);  unsqueeze = None
        scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(255, dtype = torch.int64, device = device(type='cuda', index=0), pin_memory = False)
        minimum: "i64[129, 3]" = torch.ops.aten.minimum.default(add, scalar_tensor);  add = scalar_tensor = None
        add_1: "i64[129]" = torch.ops.aten.add.Tensor(iota, 1);  iota = None
        mul_1: "i64[129]" = torch.ops.aten.mul.Tensor(add_1, 256);  add_1 = None
        add_2: "i64[129]" = torch.ops.aten.add.Tensor(mul_1, 129);  mul_1 = None
        sub: "i64[129]" = torch.ops.aten.sub.Tensor(add_2, 1);  add_2 = None
        div_1: "i64[129]" = torch.ops.aten.div.Tensor_mode(sub, 129, rounding_mode = 'trunc');  sub = None
        sub_1: "i64[129]" = torch.ops.aten.sub.Tensor(div_1, div);  div_1 = div = None
        iota_2: "i64[129]" = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2: "i64[129]" = torch.ops.aten.mul.Tensor(iota_2, 256)
        div_2: "i64[129]" = torch.ops.aten.div.Tensor_mode(mul_2, 129, rounding_mode = 'trunc');  mul_2 = None
        iota_3: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[129, 1]" = torch.ops.aten.unsqueeze.default(div_2, -1)
        add_3: "i64[129, 3]" = torch.ops.aten.add.Tensor(unsqueeze_1, iota_3);  unsqueeze_1 = None
        scalar_tensor_1: "i64[]" = torch.ops.aten.scalar_tensor.default(255, dtype = torch.int64, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "i64[129, 3]" = torch.ops.aten.minimum.default(add_3, scalar_tensor_1);  add_3 = scalar_tensor_1 = None
        add_4: "i64[129]" = torch.ops.aten.add.Tensor(iota_2, 1);  iota_2 = None
        mul_3: "i64[129]" = torch.ops.aten.mul.Tensor(add_4, 256);  add_4 = None
        add_5: "i64[129]" = torch.ops.aten.add.Tensor(mul_3, 129);  mul_3 = None
        sub_2: "i64[129]" = torch.ops.aten.sub.Tensor(add_5, 1);  add_5 = None
        div_3: "i64[129]" = torch.ops.aten.div.Tensor_mode(sub_2, 129, rounding_mode = 'trunc');  sub_2 = None
        sub_3: "i64[129]" = torch.ops.aten.sub.Tensor(div_3, div_2);  div_3 = div_2 = None
        unsqueeze_2: "i64[129, 3, 1]" = torch.ops.aten.unsqueeze.default(minimum, -1);  minimum = None
        unsqueeze_3: "i64[129, 3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        index: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.index.Tensor(args_1, [None, None, unsqueeze_3, minimum_1]);  args_1 = None
        unsqueeze_4: "i64[129, 1]" = torch.ops.aten.unsqueeze.default(sub_1, -1)
        ge: "b8[129, 3]" = torch.ops.aten.ge.Tensor(iota_1, unsqueeze_4);  iota_1 = unsqueeze_4 = None
        unsqueeze_5: "b8[129, 3, 1]" = torch.ops.aten.unsqueeze.default(ge, -1);  ge = None
        unsqueeze_6: "b8[129, 3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, -1);  unsqueeze_5 = None
        scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.where.self(unsqueeze_6, scalar_tensor_2, index);  scalar_tensor_2 = index = None
        unsqueeze_7: "i64[129, 1]" = torch.ops.aten.unsqueeze.default(sub_1, -1);  sub_1 = None
        unsqueeze_8: "i64[129, 1]" = torch.ops.aten.unsqueeze.default(sub_3, -1)
        ge_1: "b8[129, 3]" = torch.ops.aten.ge.Tensor(iota_3, unsqueeze_8);  iota_3 = unsqueeze_8 = None
        scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.where.self(ge_1, scalar_tensor_3, where);  scalar_tensor_3 = where = None
        select: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select.int(where_1, 3, 0)
        slice_1: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice.Tensor(select, 3, 0, 9223372036854775807);  select = None
        select_1: "f32[8, 3, 129, 129]" = torch.ops.aten.select.int(slice_1, 4, 0);  slice_1 = None
        select_2: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select.int(where_1, 3, 0)
        slice_2: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice.Tensor(select_2, 3, 0, 9223372036854775807);  select_2 = None
        select_3: "f32[8, 3, 129, 129]" = torch.ops.aten.select.int(slice_2, 4, 1);  slice_2 = None
        add_6: "f32[8, 3, 129, 129]" = torch.ops.aten.add.Tensor(select_1, select_3);  select_1 = select_3 = None
        select_4: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select.int(where_1, 3, 0)
        slice_3: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice.Tensor(select_4, 3, 0, 9223372036854775807);  select_4 = None
        select_5: "f32[8, 3, 129, 129]" = torch.ops.aten.select.int(slice_3, 4, 2);  slice_3 = None
        add_7: "f32[8, 3, 129, 129]" = torch.ops.aten.add.Tensor(add_6, select_5);  add_6 = select_5 = None
        select_6: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select.int(where_1, 3, 1)
        slice_4: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice.Tensor(select_6, 3, 0, 9223372036854775807);  select_6 = None
        select_7: "f32[8, 3, 129, 129]" = torch.ops.aten.select.int(slice_4, 4, 0);  slice_4 = None
        add_8: "f32[8, 3, 129, 129]" = torch.ops.aten.add.Tensor(add_7, select_7);  add_7 = select_7 = None
        select_8: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select.int(where_1, 3, 1)
        slice_5: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice.Tensor(select_8, 3, 0, 9223372036854775807);  select_8 = None
        select_9: "f32[8, 3, 129, 129]" = torch.ops.aten.select.int(slice_5, 4, 1);  slice_5 = None
        add_9: "f32[8, 3, 129, 129]" = torch.ops.aten.add.Tensor(add_8, select_9);  add_8 = select_9 = None
        select_10: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select.int(where_1, 3, 1)
        slice_6: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice.Tensor(select_10, 3, 0, 9223372036854775807);  select_10 = None
        select_11: "f32[8, 3, 129, 129]" = torch.ops.aten.select.int(slice_6, 4, 2);  slice_6 = None
        add_10: "f32[8, 3, 129, 129]" = torch.ops.aten.add.Tensor(add_9, select_11);  add_9 = select_11 = None
        select_12: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select.int(where_1, 3, 2)
        slice_7: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice.Tensor(select_12, 3, 0, 9223372036854775807);  select_12 = None
        select_13: "f32[8, 3, 129, 129]" = torch.ops.aten.select.int(slice_7, 4, 0);  slice_7 = None
        add_11: "f32[8, 3, 129, 129]" = torch.ops.aten.add.Tensor(add_10, select_13);  add_10 = select_13 = None
        select_14: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select.int(where_1, 3, 2)
        slice_8: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice.Tensor(select_14, 3, 0, 9223372036854775807);  select_14 = None
        select_15: "f32[8, 3, 129, 129]" = torch.ops.aten.select.int(slice_8, 4, 1);  slice_8 = None
        add_12: "f32[8, 3, 129, 129]" = torch.ops.aten.add.Tensor(add_11, select_15);  add_11 = select_15 = None
        select_16: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select.int(where_1, 3, 2);  where_1 = None
        slice_9: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice.Tensor(select_16, 3, 0, 9223372036854775807);  select_16 = None
        select_17: "f32[8, 3, 129, 129]" = torch.ops.aten.select.int(slice_9, 4, 2);  slice_9 = None
        add_13: "f32[8, 3, 129, 129]" = torch.ops.aten.add.Tensor(add_12, select_17);  add_12 = select_17 = None
        mul_4: "i64[129, 129]" = torch.ops.aten.mul.Tensor(unsqueeze_7, sub_3);  unsqueeze_7 = sub_3 = None
        div_4: "f32[8, 3, 129, 129]" = torch.ops.aten.div.Tensor(add_13, mul_4);  add_13 = None
        div_5: "f32[8, 3, 129, 129]" = torch.ops.aten.div.Tensor(args_2, mul_4);  args_2 = mul_4 = None
        full: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select_scatter.default(full, div_5, 4, 2);  full = None
        full_1: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice_scatter.default(full_1, select_scatter, 3, 0, 9223372036854775807);  full_1 = select_scatter = None
        full_2: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 3, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_1: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.select_scatter.default(full_2, slice_scatter, 3, 2);  full_2 = slice_scatter = None
        full_3: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_2: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select_scatter.default(full_3, div_5, 4, 1);  full_3 = None
        full_4: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_1: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice_scatter.default(full_4, select_scatter_2, 3, 0, 9223372036854775807);  full_4 = select_scatter_2 = None
        full_5: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 3, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_3: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.select_scatter.default(full_5, slice_scatter_1, 3, 2);  full_5 = slice_scatter_1 = None
        add_14: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.add.Tensor(select_scatter_1, select_scatter_3);  select_scatter_1 = select_scatter_3 = None
        full_6: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_4: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select_scatter.default(full_6, div_5, 4, 0);  full_6 = None
        full_7: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_2: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice_scatter.default(full_7, select_scatter_4, 3, 0, 9223372036854775807);  full_7 = select_scatter_4 = None
        full_8: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 3, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_5: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.select_scatter.default(full_8, slice_scatter_2, 3, 2);  full_8 = slice_scatter_2 = None
        add_15: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.add.Tensor(add_14, select_scatter_5);  add_14 = select_scatter_5 = None
        full_9: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_6: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select_scatter.default(full_9, div_5, 4, 2);  full_9 = None
        full_10: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_3: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice_scatter.default(full_10, select_scatter_6, 3, 0, 9223372036854775807);  full_10 = select_scatter_6 = None
        full_11: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 3, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_7: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.select_scatter.default(full_11, slice_scatter_3, 3, 1);  full_11 = slice_scatter_3 = None
        add_16: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.add.Tensor(add_15, select_scatter_7);  add_15 = select_scatter_7 = None
        full_12: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_8: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select_scatter.default(full_12, div_5, 4, 1);  full_12 = None
        full_13: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_4: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice_scatter.default(full_13, select_scatter_8, 3, 0, 9223372036854775807);  full_13 = select_scatter_8 = None
        full_14: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 3, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_9: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.select_scatter.default(full_14, slice_scatter_4, 3, 1);  full_14 = slice_scatter_4 = None
        add_17: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.add.Tensor(add_16, select_scatter_9);  add_16 = select_scatter_9 = None
        full_15: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_10: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select_scatter.default(full_15, div_5, 4, 0);  full_15 = None
        full_16: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_5: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice_scatter.default(full_16, select_scatter_10, 3, 0, 9223372036854775807);  full_16 = select_scatter_10 = None
        full_17: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 3, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_11: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.select_scatter.default(full_17, slice_scatter_5, 3, 1);  full_17 = slice_scatter_5 = None
        add_18: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.add.Tensor(add_17, select_scatter_11);  add_17 = select_scatter_11 = None
        full_18: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_12: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select_scatter.default(full_18, div_5, 4, 2);  full_18 = None
        full_19: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_6: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice_scatter.default(full_19, select_scatter_12, 3, 0, 9223372036854775807);  full_19 = select_scatter_12 = None
        full_20: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 3, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_13: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.select_scatter.default(full_20, slice_scatter_6, 3, 0);  full_20 = slice_scatter_6 = None
        add_19: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.add.Tensor(add_18, select_scatter_13);  add_18 = select_scatter_13 = None
        full_21: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_14: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select_scatter.default(full_21, div_5, 4, 1);  full_21 = None
        full_22: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_7: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice_scatter.default(full_22, select_scatter_14, 3, 0, 9223372036854775807);  full_22 = select_scatter_14 = None
        full_23: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 3, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_15: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.select_scatter.default(full_23, slice_scatter_7, 3, 0);  full_23 = slice_scatter_7 = None
        add_20: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.add.Tensor(add_19, select_scatter_15);  add_19 = select_scatter_15 = None
        full_24: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_16: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.select_scatter.default(full_24, div_5, 4, 0);  full_24 = div_5 = None
        full_25: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_8: "f32[8, 3, 129, 129, 3]" = torch.ops.aten.slice_scatter.default(full_25, select_scatter_16, 3, 0, 9223372036854775807);  full_25 = select_scatter_16 = None
        full_26: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.full.default([8, 3, 129, 3, 129, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_17: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.select_scatter.default(full_26, slice_scatter_8, 3, 0);  full_26 = slice_scatter_8 = None
        add_21: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.add.Tensor(add_20, select_scatter_17);  add_20 = select_scatter_17 = None
        scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_2: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.where.self(ge_1, scalar_tensor_4, add_21);  ge_1 = scalar_tensor_4 = add_21 = None
        scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_3: "f32[8, 3, 129, 3, 129, 3]" = torch.ops.aten.where.self(unsqueeze_6, scalar_tensor_5, where_2);  unsqueeze_6 = scalar_tensor_5 = where_2 = None
        full_27: "f32[8, 3, 256, 256]" = torch.ops.aten.full.default([8, 3, 256, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[8, 3, 256, 256]" = torch.ops.aten.index_put.default(full_27, [None, None, unsqueeze_3, minimum_1], where_3, True);  full_27 = unsqueeze_3 = minimum_1 = where_3 = None
        alias: "f32[8, 3, 256, 256]" = torch.ops.aten.alias.default(index_put);  index_put = None
        alias_1: "f32[8, 3, 256, 256]" = torch.ops.aten.alias.default(alias);  alias = None
        sum_1: "f32[8, 3, 129]" = torch.ops.aten.sum.dim_IntList(div_4, [-1]);  div_4 = None
        sum_2: "f32[8, 3]" = torch.ops.aten.sum.dim_IntList(sum_1, [-1]);  sum_1 = None
        sum_3: "f32[8, 3, 256]" = torch.ops.aten.sum.dim_IntList(alias_1, [-1]);  alias_1 = None
        sum_4: "f32[8, 3]" = torch.ops.aten.sum.dim_IntList(sum_3, [-1]);  sum_3 = None
        add_22: "f32[8, 3]" = torch.ops.aten.add.Tensor(sum_2, sum_4);  sum_2 = sum_4 = None
        return pytree.tree_unflatten([add_22], self._out_spec)
        