"""Minimal reader for ``.safetensors`` files, so the package does not depend on the ``safetensors`` library.

Format: 8 bytes little-endian header size, a JSON header ``{name: {dtype, shape, data_offsets}, "__metadata__": {}}``,
then the raw little-endian tensor bytes.
"""
from __future__ import annotations

import json
import os
import sys

import torch

_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool,
}


def load_file(path: str | os.PathLike) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read every tensor of a safetensors file into CPU tensors. Returns (tensors, metadata)."""
    if sys.byteorder != "little":
        raise RuntimeError("safetensors files are little-endian; big-endian machines are not supported")
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if not 0 < n < 100_000_000:
            raise ValueError(f"{path}: not a safetensors file")
        header = json.loads(f.read(n))
        data = bytearray(f.read())
    meta = header.pop("__metadata__", None) or {}
    tensors = {}
    for name, info in header.items():
        dtype = _DTYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"{path}: unsupported dtype {info['dtype']} for {name}")
        start, end = info["data_offsets"]
        shape = info["shape"]
        if end == start:
            tensors[name] = torch.empty(shape, dtype=dtype)
            continue
        t = torch.frombuffer(data, dtype=dtype, count=(end - start) // dtype.itemsize, offset=start)
        tensors[name] = t.reshape(shape)
    return tensors, meta
