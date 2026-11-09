"""Owned CPU float32 model snapshots for direct MPI buffer transfers."""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch


@dataclass(frozen=True)
class ModelBuffer:
    values: np.ndarray
    layout: tuple

    @property
    def layout_id(self):
        return hashlib.sha256(json.dumps(self.layout, separators=(',', ':')).encode()).hexdigest()

    @property
    def digest(self):
        return hashlib.sha256(self.values.tobytes()).hexdigest()

    def state_dict(self):
        """Return independent tensors; neither received nor send buffers are aliased."""
        result, offset = {}, 0
        for name, shape in self.layout:
            count = math.prod(shape)
            result[name] = torch.from_numpy(self.values[offset:offset+count].copy()).reshape(shape)
            offset += count
        if offset != self.values.size:
            raise ValueError('Model layout does not match buffer size')
        return result


def pack_state(state):
    """Never coerce unsupported types or expose a live model as a send buffer."""
    if not state or any(not isinstance(name, str) for name in state):
        raise ValueError('A nonempty named state dictionary is required')
    arrays, layout = [], []
    for name in sorted(state):
        tensor = state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                or tensor.dtype != torch.float32 or tensor.layout != torch.strided
                or tensor.numel() == 0):
            raise ValueError(f'{name}: only nonempty dense CPU float32 tensors are supported')
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f'{name}: snapshot contains non-finite values')
        arrays.append(tensor.detach().contiguous().numpy().reshape(-1))
        layout.append((name, tuple(tensor.shape)))
    values = np.concatenate(arrays).copy()
    values.flags.writeable = False
    return ModelBuffer(values, tuple(layout))
