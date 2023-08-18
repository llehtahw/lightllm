import numpy as np
from multiprocessing import Queue
import multiprocessing
import contextlib


@contextlib.contextmanager
def patch():
    import triton.runtime.jit
    from triton.compiler import CompiledKernel as CompiledKernel
    import functools
    from torch.cuda import current_device, current_stream
    from torch import Tensor
    orig_JF_key_of = triton.runtime.jit.JITFunction._key_of
    orig_JF_init = triton.runtime.jit.JITFunction.__init__
    @staticmethod
    def replace_JF_key_of(arg):
        tp = arg.__class__
        if tp is int:
            if -2**31 <= arg <= 2**31 - 1:
                return "i32"
            elif 2**63 <= arg <= 2**64 - 1:
                return "u64"
            else:
                return "i64"
        elif tp is Tensor:
            return arg.dtype
        elif tp is bool:
            return "i1"
        elif tp is float:
            return 'fp32'
        elif arg is None:
            return None
        elif hasattr(arg, "dtype"):
            print('#deg', type(arg))
            return arg.dtype
        else:
            raise TypeError(f'Unsupported type {type(arg)} for {arg}')
    def replace_JF_init(self, *args, **kwargs):
        orig_JF_init(self, *args, **kwargs)
        self.run = functools.partial(self.run, device_type='cuda', stream=current_stream().cuda_stream, device=current_device())
    orig_CK_assemble_tenosrmap_to_arg = CompiledKernel.assemble_tensormap_to_arg
    orig_CK_init = CompiledKernel.__init__
    orig_CK_getattribute = CompiledKernel.__getattribute__
    #@property
    def replace_CK_getattr(self, name):
        if name == 'c_wrapper':
            self._init_handles()
            self.c_wrapper = self._c_wrapper
            return self.c_wrapper
        raise AttributeError(f"{self!r} has no attribute {name!r}")
    def replace_CK_init(self, *args, **kwargs):
        orig_CK_init(self, *args, **kwargs)
        self.has_tensormaps_info = hasattr(self, 'tensormaps_info')
        self._c_wrapper = self.c_wrapper
        del self.c_wrapper
    from torch import Tensor
    def replace_CK_assemble_tensormap_to_arg(self, args):
        args_with_tma = args
        if self.has_tensormaps_info:
            if not args.__class__ is list:
                args_with_tma = list(args)
            args_ptr = tuple([arg.data_ptr() if arg.__class__ is Tensor else arg for arg in args])
            #for e in self.tensormaps_info:
            #    args_with_tma.append(CompiledKernel.tensormap_manager[(e, args_ptr)])
            args_with_tma.extend(map(lambda e : CompiledKernel.tensormap_manager[(e, args_ptr)], self.tensormaps_info))
        return args_with_tma
    try:
        triton.runtime.jit.JITFunction._key_of = replace_JF_key_of
        triton.runtime.jit.JITFunction.__init__ = replace_JF_init
        CompiledKernel.assemble_tensormap_to_arg = replace_CK_assemble_tensormap_to_arg
        CompiledKernel.__init__ = replace_CK_init
        CompiledKernel.__getattr__ = replace_CK_getattr
        del CompiledKernel.__getattribute__
        print('patch done')
        yield
    finally:
        CompiledKernel.__getattribute__ = orig_CK_getattribute
        if '__getattr__' in CompiledKernel.__dict__:
            del CompiledKernel.__getattr__
        CompiledKernel.__init__ = orig_CK_init
        CompiledKernel.assemble_tensormap_to_arg = orig_CK_assemble_tenosrmap_to_arg
        triton.runtime.jit.JITFunction.__init__ = orig_JF_init
        triton.runtime.jit.JITFunction._key_of = orig_JF_key_of


def test_model_inference(world_size, model_dir, model_class, batch_size, input_len, output_len, use_patch=False):
    ans_queue = Queue()
    workers = []
    launch_fn = tppart_model_infer
    if use_patch:
        launch_fn = tppart_model_infer_wrap
    for rank_id in range(world_size):
        proc = multiprocessing.Process(target=launch_fn, args=(rank_id, world_size, ans_queue, model_dir, model_class, batch_size, input_len, output_len))
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    assert not ans_queue.empty()
    while not ans_queue.empty():
        assert ans_queue.get()
    return

def tppart_model_infer_wrap(*args):
    with patch():
        return tppart_model_infer(*args)

def tppart_model_infer(rank_id, world_size, ans_queue, model_dir, model_class, batch_size, input_len, output_len):
    import importlib
    if isinstance(model_class, str):
        pos = model_class.rfind('.')
        model_class = getattr(importlib.import_module(model_class[:pos]), model_class[pos+1:])
    import torch
    import torch.distributed as dist
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id)

    import torch.distributed as dist
    dist.barrier()
    torch.cuda.empty_cache()

    model_part = model_class(dist.get_rank(),
                             dist.get_world_size(),
                             max_total_token_num= batch_size * (input_len + output_len),
                             weight_dir=model_dir,
                             load_way="HF")
    # warm up
    test_data = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data).cuda()

    b_loc = torch.zeros(batch_size, input_len + output_len, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_loc[i, 0:input_len] = i * input_len + torch.arange(0, input_len, dtype=torch.int32, device="cuda")
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = input_len * batch_size
    logics = model_part.forward(batch_size,
                                total_token_num,
                                input_len,
                                test_data,
                                b_loc,
                                b_start_loc,
                                b_seq_len,
                                is_prefill=True)
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    for i in range(output_len):
        b_loc[:, input_len + i] = total_token_num + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1
        logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_loc, b_start_loc, b_seq_len, is_prefill=False)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

    max_len_in_batch = input_len + output_len
    for i in range(batch_size):
        model_part.mem_manager.free(b_loc[i, max_len_in_batch - b_seq_len[i]:max_len_in_batch])
    if rank_id == 0:
        print("can use mem size:", model_part.mem_manager.can_use_mem_size)

    b_loc = None
    b_start_loc = None
    b_seq_len = None

    dist.barrier()
    import time
    torch.cuda.synchronize()
    start_time = time.time()

    prefill_start_time = time.time()

    b_loc = torch.zeros(batch_size, input_len + output_len, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = batch_size * input_len
    logics = model_part.forward(batch_size, total_token_num, input_len, test_data,
                                                 b_loc, b_start_loc, b_seq_len, is_prefill=True)
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    torch.cuda.synchronize()
    if rank_id == 0:
        print("prefill time cost:", (time.time() - prefill_start_time) * 1000)

    for i in range(output_len):
        torch.cuda.synchronize()
        step_start = time.time()
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1

        logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_loc, b_start_loc, b_seq_len, is_prefill=False)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        torch.cuda.synchronize()
        if i % 100 == 0 or i == output_len - 1:
            if rank_id == 0:
                print(i, "step cost time:", (time.time() - step_start) * 1000)

    torch.cuda.synchronize()
    end_time = time.time()

    if rank_id == 0:
        print("time total cost(ms):", (end_time - start_time) * 1000)
    ans_queue.put(True)

    return


