"""Concept: filaments. Every word's journey through the 28 layers, where each step is drawn as the real sum it is.

A layer moves a word by attn + mlp. The attention part is a sum of 16 heads x (every earlier word) contributions,
P[h,t,s] * Wo_h v_s; the MLP part is a sum of 6144 neuron contributions, act_j * Wdown[:, j]. Drawn tip to tail in a
fixed 3D projection, those contributions trace a filament from where the word was to where it goes: thousands of
pushes that mostly cancel, leaving the small net step. Rendered by additive splatting (like the GPU would).
usage: concept_filaments.py [word index or -1 for all prompt words]
"""
import sys, os
import numpy as np
import torch
from PIL import Image
from flow_common import load, capture

which = int(sys.argv[1]) if len(sys.argv) > 1 else -1
tok, model = load()
C = capture(tok, model)
words = C['words']
W = list(range(3, C['end']))
nl = C['act'].shape[0]
H, HD = 16, 128

# fixed 3D basis: PCA of the prompt words' states over all layers, each layer rescaled to a common size
scale = torch.stack([C['resid'][l][W].norm(dim=1).median() for l in range(nl + 1)]).double()
X = torch.cat([C['resid'][l][W].double() / scale[l] for l in range(1, nl + 1)])
mu = X.mean(0)
_, _, Vt = torch.linalg.svd(X - mu, full_matrices=False)
E = Vt[:3].T  # D x 3
proj = lambda v, l: ((v.double() / scale[l]) @ E).numpy()

pts, cols = [], []
MINUS = np.array([0.093, 0.578, 1.0]); PLUS = np.array([1.0, 0.434, 0.069]); BONE = np.array([0.807, 0.761, 0.672])
hues = np.array([np.array([0.5 + 0.5 * np.cos(2 * np.pi * (k / len(W) + o)) for o in (0, 0.33, 0.67)]) for k in range(len(W))])
for wi, t in enumerate(W if which < 0 else [which]):
    for l in range(nl):
        L = model.model.layers[l]
        Wo = L.self_attn.o_proj.weight.float().cpu().double()  # D x (H*HD)
        Wd = L.mlp.down_proj.weight.float().cpu().double()  # D x F
        start = ((C['resid'][l][t].double() - mu * scale[l]) / scale[l]) @ E
        # attention: one contribution per (source word, head)
        steps, sc = [], []
        v = C['v'][l].double()  # n x (8*128)
        for s in range(t + 1):
            for h in range(H):
                p = C['probs'][l][h, t, s].item()
                if p < 1e-3:
                    continue
                vec = Wo[:, h * HD:(h + 1) * HD] @ v[s, (h // 2) * HD:(h // 2 + 1) * HD] * p
                steps.append((vec / scale[l]) @ E)
                sc.append(BONE * 0.5 if s < 3 or s == t else hues[W.index(s)] if s in W else BONE * 0.3)
        # MLP: one contribution per neuron, in the model's own neuron order
        a = C['act'][l][t].double()
        contrib = ((Wd * a[None]).T / scale[l]) @ E  # F x 3
        steps += list(contrib)
        base = hues[W.index(t)] * 0.7 + 0.3
        sc += [base * (0.75 + 0.25 * (PLUS if x > 0 else MINUS)) for x in a.numpy()]
        path = start.numpy() + np.cumsum(np.stack([s.numpy() for s in steps]), 0)
        pts.append(np.vstack([start.numpy()[None], path]))
        cols.append(np.vstack([BONE[None], np.array(sc)]))
    print('word', repr(words[t]), 'done')

# additive splat render from a few viewpoints
def render(name, yaw, pitch, size=2000, gain=1.0):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    rot = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    allq = np.vstack(pts) @ rot.T
    lo, hi = np.percentile(allq[:, :2], 0.3, axis=0), np.percentile(allq[:, :2], 99.7, axis=0)
    c2 = (lo + hi) / 2
    R = (hi - lo).max() / 2 * 1.08
    img = np.zeros((size, size, 3))
    for P, Cc in zip(pts, cols):
        q = P @ rot.T
        seg = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1) / R * size * 0.5
        k = np.maximum(1, np.ceil(seg * 1.5).astype(int))
        idx = np.repeat(np.arange(len(q) - 1), k)
        f = np.concatenate([np.arange(n) / n for n in k])
        qq = q[idx] + (q[idx + 1] - q[idx]) * f[:, None]
        cc = Cc[idx + 1] / np.repeat(k, k)[:, None] ** 0.3
        x = ((qq[:, 0] - c2[0]) / R * 0.5 + 0.5) * size
        y = (0.5 - (qq[:, 1] - c2[1]) / R * 0.5) * size
        ok = (x >= 0) & (x < size - 1) & (y >= 0) & (y < size - 1)
        x, y, cc = x[ok], y[ok], cc[ok]
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(img, (y0 + dy, x0 + dx), cc * w[:, None] * 0.05 * gain)
    from scipy.ndimage import gaussian_filter
    glow = np.stack([gaussian_filter(img[..., i], 6) for i in range(3)], -1)
    out = 1 - np.exp(-(img + glow * 2.0) * 2.5)
    Image.fromarray((np.clip(out, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)).save(f'renders/flow/{name}.png')

os.makedirs('renders/flow', exist_ok=True)
tag = 'all' if which < 0 else words[which].strip()
render(f'filaments_{tag}_a', 0.0, 0.0)
render(f'filaments_{tag}_b', 1.1, 0.4)
