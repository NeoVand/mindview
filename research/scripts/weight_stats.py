"""Verify that unpacked Bonsai weights are exactly {-1,0,+1}*scale per 128-group and gather statistics."""
import sys, glob, json
import numpy as np
from safetensors import safe_open

repo = sys.argv[1]  # e.g. Ternary-Bonsai-1.7B-unpacked
path = glob.glob(f'/Users/neo/.cache/huggingface/hub/models--prism-ml--{repo}/snapshots/*/model.safetensors')[0]
G = 128
out = {}
with safe_open(path, 'np') as f:
    keys = list(f.keys())
    print(len(keys), 'tensors')
    for k in keys:
        w = f.get_tensor(k)
        if w.ndim != 2:
            continue
        w32 = w.astype(np.float32)
        rows, cols = w32.shape
        g = w32.reshape(rows, cols // G, G)
        scale = np.abs(g).max(axis=2, keepdims=True)
        scale[scale == 0] = 1
        t = g / scale
        tr = np.rint(t)
        err = np.abs(t - tr).max()
        vals, counts = np.unique(tr, return_counts=True)
        frac = {int(v): c / tr.size for v, c in zip(vals, counts)}
        s = scale.squeeze(-1)
        out[k] = dict(shape=[rows, cols], max_dev=float(err), frac=frac,
                      scale_mean=float(s.mean()), scale_std=float(s.std()), scale_min=float(s.min()), scale_max=float(s.max()))
        if 'layers.0.' in k or 'layers.27.' in k or 'embed' in k or 'lm_head' in k or 'layers.14.' in k:
            print(f"{k:50s} {str(w.shape):14s} dev={err:.2e} " + ' '.join(f"{v:+d}:{p*100:5.1f}%" for v, p in frac.items()) + f"  scale μ={s.mean():.4f} σ={s.std():.4f}")
json.dump(out, open(f'data/{repo}.weightstats.json', 'w'), indent=1)
zeros = [v['frac'].get(0, 0) for v in out.values()]
print('mean zero fraction over matrices:', np.mean(zeros), 'min', np.min(zeros), 'max', np.max(zeros))
print('max deviation from exact ternary:', max(v['max_dev'] for v in out.values()))
