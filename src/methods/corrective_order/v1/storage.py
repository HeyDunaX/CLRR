"""Exact saved-tensor disk spill within the author's project; bounded resources."""
from contextlib import contextmanager
import math
from pathlib import Path
import shutil
import tempfile
import weakref

import torch


class SavedTensorFiles:
    def __init__(self, root, minimum_bytes=2**20, reserve_bytes=6*2**30, disk_reserve=40*2**30):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='graph-', dir=root)
        self.root = Path(self.directory.name).resolve()
        if not self.root.is_relative_to(root):
            raise ValueError('Saved tensor directory escaped project')
        self.minimum_bytes, self.reserve_bytes, self.disk_reserve = minimum_bytes, reserve_bytes, disk_reserve
        self.cache = {}
        self.count = 0

    def pack(self, tensor):
        size = tensor.numel()*tensor.element_size()
        if size < self.minimum_bytes or not tensor.numel():
            return tensor.device, tensor.detach()
        key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor._version)
        cached = self.cache.get(key)
        if cached is not None and cached[0]() is tensor:
            return cached[1]
        if shutil.disk_usage(self.root).free < self.disk_reserve+size:
            raise RuntimeError('DISK_ADMISSION_FAILED: preserve 40 GiB; no protocol changes')
        if tensor.is_cuda and Path('/proc/meminfo').exists():
            available = next(int(line.split()[1])*1024 for line in
                Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
            if available < self.reserve_bytes+size:
                raise RuntimeError('HOST_RAM_ADMISSION_FAILED during bounded disk transfer')
        path = self.root/f'{self.count}.bin'
        self.count += 1
        # Synchronous write: never retain a CPU copy for the graph lifetime.
        tensor.detach().to('cpu').contiguous().numpy().tofile(path)
        saved = (tensor.device, (str(path), tensor.dtype, tuple(tensor.shape)))
        self.cache[key] = (weakref.ref(tensor), saved)
        return saved

    def unpack(self, saved):
        device, payload = saved
        if isinstance(payload, torch.Tensor):
            return payload.to(device)
        name, dtype, shape = payload
        mapped = torch.from_file(name, shared=False, size=math.prod(shape), dtype=dtype)
        # copy=True also releases Windows mmap handles for CPU correctness tests.
        return mapped.reshape(shape).to(device=device, copy=True)


@contextmanager
def disk_saved_tensors(root, **kwargs):
    store = SavedTensorFiles(root, **kwargs)
    # The saved unpack bound method retains store/directory until its graph is
    # freed, INCLUDING backward after this forward context has exited.
    with torch.autograd.graph.saved_tensors_hooks(store.pack, store.unpack):
        yield store
