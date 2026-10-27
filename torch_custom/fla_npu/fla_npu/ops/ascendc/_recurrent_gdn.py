"""Recurrent GDN structural and metadata checks shared by both Python backends."""
from __future__ import annotations

import math
import numbers
import weakref
from collections import OrderedDict

_METADATA_CACHE = OrderedDict()
_SOC_CACHE = {}


def _storage_bounds(name, tensor):
    shape, stride = tuple(tensor.shape), tuple(tensor.stride())
    if any(s <= 0 for s in stride):
        raise ValueError(f"{name}: strides must be positive")
    last = tensor.storage_offset() + sum((n - 1) * s for n, s in zip(shape, stride))
    if tensor.storage_offset() < 0 or last >= tensor.untyped_storage().nbytes() // tensor.element_size():
        raise ValueError(f"{name}: view exceeds storage bounds")


def _metadata(lengths, indices, accepted, total_tokens, blocks):
    tensors = (lengths, indices, accepted)
    versions = []
    for tensor in tensors:
        try:
            versions.append(None if tensor is None else tensor._version)
        except RuntimeError:
            versions.append(None)
    key = (tuple(id(t) for t in tensors), tuple(versions), total_tokens, blocks)
    cached = _METADATA_CACHE.get(key)
    if cached is not None and all(ref is None if t is None else ref() is t
                                  for ref, t in zip(cached, tensors)):
        _METADATA_CACHE.move_to_end(key)
        return
    lens = lengths.detach().cpu().tolist()
    if any(n < 0 for n in lens) or sum(lens) != total_tokens:
        raise ValueError("actual_seq_lengths must be nonnegative and sum to T")
    table = indices.detach().cpu().tolist()
    flat = table if indices.ndim == 1 else [slot for row in table for slot in row]
    if any(slot < 0 or slot >= blocks for slot in flat):
        raise ValueError("ssm_state_indices values must be in [0, BlockNum)")
    accepted_list = [1] * (len(lens) - 1) if accepted is None else accepted.detach().cpu().tolist()
    writers, readers = {}, []
    start = lens[0]
    for b, length in enumerate(lens[1:]):
        width = length if indices.ndim == 1 else indices.shape[1]
        if length > width:
            raise ValueError("sequence length exceeds ssm_state_indices table width")
        if length:
            if not 1 <= accepted_list[b] <= width:
                raise ValueError("num_accepted_tokens checkpoint out of range")
            row = table[start:start + length] if indices.ndim == 1 else table[b]
            readers.append((row[accepted_list[b] - 1], b))
            for slot in row[:length]:
                owner = writers.setdefault(slot, b)
                if owner != b:
                    raise ValueError("State writes overlap across requests")
        start += length
    for slot, b in readers:
        if slot in writers and writers[slot] != b:
            raise ValueError("initial State races another request")
    # Inference tensors do not carry mutation versions and cannot be cached.
    if all(t is None or version is not None for t, version in zip(tensors, versions)):
        _METADATA_CACHE[key] = tuple(None if t is None else weakref.ref(t) for t in tensors)
        if len(_METADATA_CACHE) > 64:
            _METADATA_CACHE.popitem(last=False)


def validate_recurrent_gdn(query, key, value, state, *, beta, scale,
                           actual_seq_lengths, ssm_state_indices, num_accepted_tokens=None,
                           g=None, gk=None, A_log=None, dt_bias=None,
                           use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                           use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                           _allow_cpu=False):
    import torch

    if gk is not None:
        raise ValueError("gk is reserved; only scalar gate g is supported")
    switches = (use_qk_l2norm_in_kernel, use_gate_in_kernel,
                use_beta_sigmoid_in_kernel, allow_neg_eigval)
    if any(not isinstance(flag, bool) for flag in switches):
        raise TypeError("recurrent GDN switches must be bool")
    if not isinstance(scale, numbers.Real) or isinstance(scale, bool) or not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    required = dict(query=query, key=key, value=value, state=state, beta=beta,
                    actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices)
    optional = dict(num_accepted_tokens=num_accepted_tokens, g=g, A_log=A_log, dt_bias=dt_bias)
    expected_dtypes = dict(query=(torch.bfloat16,), key=(torch.bfloat16,), value=(torch.bfloat16,),
        state=(torch.bfloat16, torch.float32), beta=(torch.bfloat16,),
        actual_seq_lengths=(torch.int32,), ssm_state_indices=(torch.int32,),
        num_accepted_tokens=(torch.int32,), g=(torch.float32,), A_log=(torch.float32,), dt_bias=(torch.float32,))
    device = None
    for name, tensor in {**required, **optional}.items():
        if tensor is None and name in optional:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != ("cpu" if _allow_cpu else "npu"):
            raise TypeError(f"{name} must be on {'CPU' if _allow_cpu else 'NPU'}")
        if device is None:
            device = tensor.device
        if tensor.device != device:
            raise ValueError("all inputs must be on the same device")
        if tensor.dtype not in expected_dtypes[name]:
            raise TypeError(f"{name}: unsupported dtype {tensor.dtype}")
        _storage_bounds(name, tensor)
    if query.ndim != 3 or key.shape != query.shape or value.ndim != 3 or state.ndim != 4:
        raise ValueError("query/key/value/state rank or shape mismatch")
    total_tokens, nk, kdim = query.shape
    nv, vdim = value.shape[1:]
    if total_tokens <= 0 or nk <= 0 or nv <= 0 or state.shape[0] <= 0:
        raise ValueError("T, Nk, Nv and BlockNum must be positive")
    if kdim != 128 or vdim != 128 or nv % nk:
        raise ValueError("K=V=128 and Nv divisible by Nk are required")
    if value.shape[0] != total_tokens or beta.shape != (total_tokens, nv):
        raise ValueError("value token count or beta shape mismatch")
    if state.shape[1:] != (nv, 128, 128):
        raise ValueError("State shape mismatch")
    if state.stride(3) != 1 or state.stride(2) < 128:
        raise ValueError("State inner axis must be contiguous")
    span = 1
    for stride, size in sorted((s, n) for s, n in zip(state.stride(), state.shape) if n > 1):
        if stride < span:
            raise ValueError("State view must not overlap")
        span += (size - 1) * stride
    if actual_seq_lengths.ndim != 1 or actual_seq_lengths.numel() < 2:
        raise ValueError("actual_seq_lengths must be [prefix, L0, ...]")
    batch = actual_seq_lengths.numel() - 1
    if ssm_state_indices.ndim == 1:
        if ssm_state_indices.shape != (total_tokens,):
            raise ValueError("ssm_state_indices length must equal T")
    elif ssm_state_indices.ndim == 2:
        if ssm_state_indices.shape[0] != batch or ssm_state_indices.shape[1] <= 0:
            raise ValueError("ssm_state_indices table must be [B,W] with W>0")
    else:
        raise ValueError("ssm_state_indices must have rank 1 or 2")
    for name, tensor, shape in (("g", g, (total_tokens, nv)),
        ("A_log", A_log, (nv,)), ("dt_bias", dt_bias, (nv,)),
        ("num_accepted_tokens", num_accepted_tokens, (batch,))):
        if tensor is not None and tensor.shape != shape:
            raise ValueError(f"{name} shape must be {shape}")
    if use_gate_in_kernel and (g is None or A_log is None):
        raise ValueError("use_gate_in_kernel requires g and A_log")
    if not hasattr(actual_seq_lengths, "fake_mode"):
        _metadata(actual_seq_lengths, ssm_state_indices, num_accepted_tokens, total_tokens, state.shape[0])
    if not _allow_cpu:
        index = device.index if device.index is not None else torch.npu.current_device()
        is_a5 = _SOC_CACHE.get(index)
        if is_a5 is None:
            is_a5 = "950" in torch.npu.get_device_name(index)
            _SOC_CACHE[index] = is_a5
        if not is_a5:
            if any(switches) or ssm_state_indices.ndim == 2 or nk > 256 or nv > 256:
                raise NotImplementedError("extended Recurrent GDN requires Ascend950")
            if state.stride(2) != 128 or max(actual_seq_lengths.detach().cpu().tolist()[1:]) > 8:
                raise NotImplementedError("legacy Recurrent GDN requires row-contiguous State and L<=8")
    return float(scale)
