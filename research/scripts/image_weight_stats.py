"""Check which Bonsai Image tensors are ternary/binary and gather sparsity stats."""
import sys, glob
import numpy as np, torch
from safetensors import safe_open
variant = sys.argv[1]
sub = sys.argv[2] if len(sys.argv) > 2 else 'transformer'
paths = sorted(glob.glob(f'/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-{variant}-4B-unpacked/snapshots/*/{sub}/*.safetensors'))
tot = {'lowbit': 0, 'other': 0}; zeros = []
for p in paths:
    with safe_open(p, 'pt') as f:
        for k in f.keys():
            w = f.get_tensor(k)
            n = w.numel()
            if w.ndim != 2 or w.shape[1] % 128:
                tot['other'] += n; continue
            g = w.float().reshape(w.shape[0], -1, 128)
            s = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
            t = g / s
            dev = (t - t.round()).abs().max().item()
            uniq = t.round().unique().tolist()
            ok = dev < 0.02 and len(uniq) <= 3
            tot['lowbit' if ok else 'other'] += n
            if ok: zeros.append((t.round() == 0).float().mean().item())
            if not ok or 'single_transformer_blocks.10.' in k or 'transformer_blocks.2.' in k and 'single' not in k:
                print(f"{'LOWBIT' if ok else 'full  '} {k:60s} {tuple(w.shape)} {str(w.dtype):14s} dev={dev:.3f} vals={uniq[:5] if len(uniq)<=5 else len(uniq)}" + (f" zero={zeros[-1]*100:.1f}%" if ok else ''))
print(tot, 'lowbit share', tot['lowbit'] / (tot['lowbit'] + tot['other']))
if zeros: print('mean zero fraction', np.mean(zeros), 'range', np.min(zeros), np.max(zeros))
