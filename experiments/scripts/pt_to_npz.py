"""Convert OGB split .pt files (dicts of tensors or numpy arrays) to .npz, loading with
weights_only=True so that no pickled code from the download can run. Only numpy's
array-reconstruction globals are allow-listed.

usage: python pt_to_npz.py <in.pt> <out.npz>
"""
import _codecs
import sys

import numpy as np
import torch

# _codecs.encode only turns the pickled latin-1 string back into the array's bytes
safe = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, _codecs.encode]
safe += [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith('DType')]
torch.serialization.add_safe_globals(safe)
obj = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
arrays = {k: (v.numpy() if hasattr(v, 'numpy') else np.asarray(v)) for k, v in obj.items()}
np.savez(sys.argv[2], **arrays)
print(sys.argv[1], {k: (v.shape, str(v.dtype)) for k, v in arrays.items()})
