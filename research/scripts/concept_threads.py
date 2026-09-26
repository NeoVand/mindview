"""Concept: threads through maps. The 28 layers as stacked maps of meaning; each word a thread falling through them.

- Each plate is a plane through meaning-space at one layer, its regions named by the model's own vocabulary (argmax of
  the dot product with every vocabulary row). The same 2D directions are used for every plate.
- Between plate l and l+1 a word's thread descends while its step is summed: first the attention contributions (one per
  head and earlier word), then the 6144 neuron pushes. Height is how far through that sum we are; the sideways position
  is the running sum itself, so the thread lands exactly where the word is on the next plate.
- Attention routes arc between the words on each plate.
Rendered by additive splatting from an oblique camera; labels composited with PIL.
"""
import os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from flow_common import load, capture, logit_lens

tok, model = load()
C = capture(tok, model)
words = C['words']
W = list(range(3, C['end']))
nl = C['act'].shape[0]
H, HD = 16, 128
GRID = int(os.environ.get('GRID', 72))

s = torch.stack([C['resid'][l][W].norm(dim=1).median() for l in range(nl + 1)]).double()
mu = torch.stack([(C['resid'][l][W].double() / s[l]).mean(0) for l in range(nl + 1)])
X = torch.cat([C['resid'][l][W].double() / s[l] - mu[l] for l in range(1, nl + 1)])
_, _, Vt = torch.linalg.svd(X, full_matrices=False)
E = Vt[:2].T  # D x 2
pos = lambda h, l: ((h.double() / s[l] - mu[l]) @ E).numpy()

# plate extent from where the words go
allxy = np.concatenate([pos(C['resid'][l][W], l) for l in range(nl + 1)])
lo, hi = np.percentile(allxy, 1, 0), np.percentile(allxy, 99, 0)
c2, R = (lo + hi) / 2, (hi - lo).max() / 2 * 1.15
DY = R * float(os.environ.get('DY', 0.1))  # plate spacing

pts, cols = [], []  # 3D points (x, y=height, z) and colours, drawn as polylines
MINUS = np.array([0.093, 0.578, 1.0]); PLUS = np.array([1.0, 0.434, 0.069]); BONE = np.array([0.807, 0.761, 0.672])

def to3(xy, height):
    xy = np.atleast_2d(xy)
    return np.stack([xy[:, 0] - c2[0], np.full(len(xy), height), xy[:, 1] - c2[1]], 1)

# ---- plates: region boundaries from the logit lens on a grid
g = np.linspace(-R, R, GRID)
A, B = np.meshgrid(g + c2[0], g + c2[1])
plates = []
for l in range(0, nl + 1):
    P = (mu[l][None] + torch.from_numpy(A.reshape(-1, 1)) * E[:, 0][None] + torch.from_numpy(B.reshape(-1, 1)) * E[:, 1][None])
    ids, conf = logit_lens(model, P)
    ids = ids.numpy().reshape(GRID, GRID)
    plates.append(ids)
    edge = (np.diff(ids, axis=0, prepend=ids[:1]) != 0) | (np.diff(ids, axis=1, prepend=ids[:, :1]) != 0)
    yy, xx = np.nonzero(edge)
    for y_, x_ in zip(yy, xx):
        a = np.array([A[y_, x_], B[y_, x_]])
        d = R / GRID * 0.5
        pts.append(to3(np.array([a - [d, 0], a + [d, 0]]), -l * DY))
        cols.append(np.tile(BONE * 0.35, (2, 1)))
    # plate outline
    sq = np.array([[-R, -R], [R, -R], [R, R], [-R, R], [-R, -R]]) + c2
    pts.append(to3(sq, -l * DY)); cols.append(np.tile(BONE * 0.25, (5, 1)))
    print('plate', l, len(np.unique(ids)), 'regions')

# ---- threads
hues = np.array([[0.5 + 0.5 * np.cos(2 * np.pi * (k / len(W) + o)) for o in (0, 0.33, 0.67)] for k in range(len(W))])
Eo_cache = {}
for wi, t in enumerate(W):
    hue = hues[wi] * 0.6 + 0.4
    for l in range(nl):
        L = model.model.layers[l]
        Wo = L.self_attn.o_proj.weight.float().cpu().double()
        Wd = L.mlp.down_proj.weight.float().cpu().double()
        v = C['v'][l].double()
        steps, sc = [], []
        for src in range(t + 1):
            for h in range(H):
                p = C['probs'][l][h, t, src].item()
                if p < 2e-3:
                    continue
                steps.append(Wo[:, h * HD:(h + 1) * HD] @ v[src, (h // 2) * HD:(h // 2 + 1) * HD] * p)
                sc.append(hues[W.index(src)] if src in W else BONE * 0.4)
        na = len(steps)
        a = C['act'][l][t].double()
        steps = torch.stack(steps) if steps else torch.zeros(0, Wd.shape[0], dtype=torch.float64)
        partial = torch.cat([steps.cumsum(0), steps.sum(0)[None] + (Wd * a[None]).T.cumsum(0)])
        sc += [hue * (0.8 + 0.2 * (PLUS if x > 0 else MINUS)) for x in a.numpy()]
        K = len(partial)
        # height: attention takes the first 30% of the descent, the neurons the rest
        frac = np.concatenate([np.linspace(0, 0.3, na + 1)[1:], np.linspace(0.3, 1, K - na + 1)[1:]])
        h0 = C['resid'][l][t].double()
        sl = s[l] + (s[l + 1] - s[l]) * torch.from_numpy(frac)[:, None]
        ml = mu[l] + (mu[l + 1] - mu[l]) * torch.from_numpy(frac)[:, None]
        xy = (((h0[None] + partial) / sl - ml) @ E).numpy()
        xy = np.vstack([pos(h0, l)[None], xy])
        hgt = -(l + np.concatenate([[0], frac])) * DY
        pts.append(np.stack([xy[:, 0] - c2[0], hgt, xy[:, 1] - c2[1]], 1))
        cols.append(np.vstack([hue[None], np.array(sc)]) * float(os.environ.get('TG', 0.12)))
    print('thread', repr(words[t]))

# ---- attention routes on each plate
for l in range(nl):
    Pm = C['probs'][l].mean(0).numpy()
    xy = pos(C['resid'][l + 1][W], l + 1)
    for i, t in enumerate(W):
        for j, src in enumerate(W):
            if src >= t or Pm[t, src] < 0.03:
                continue
            a, b = xy[i], xy[j]
            tt = np.linspace(0, 1, 40)[:, None]
            curve = (1 - tt) * a + tt * b
            arc = np.sin(np.pi * tt[:, 0]) * np.linalg.norm(b - a) * 0.35
            p3 = to3(curve, -(l + 1) * DY)
            p3[:, 1] += arc
            pts.append(p3); cols.append(np.tile(PLUS * min(1.0, Pm[t, src] * 4), (40, 1)))

import pickle
pickle.dump((pts, cols, nl, DY, R), open('data/threads_scene.pkl', 'wb'))

def render(name, yaw, pitch, size=2200, zoom=1.0, gain=1.0):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    rot = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    allq = np.vstack(pts)
    # the whole stack is about R wide and nl*DY tall: scale so both fit
    ctr = np.array([0, -nl * DY / 2, 0])
    q_all = (allq - ctr) @ rot.T
    ext = max(np.percentile(np.abs(q_all[:, 0]), 99.5), np.percentile(np.abs(q_all[:, 1]), 99.5)) * 1.05 / zoom
    img = np.zeros((size, size, 3))
    for P, Cc in zip(pts, cols):
        q = (P - ctr) @ rot.T
        seg = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1) / ext * size * 0.5
        k = np.maximum(1, np.ceil(seg * 1.5).astype(int))
        idx = np.repeat(np.arange(len(q) - 1), k)
        f = np.concatenate([np.arange(n) / n for n in k])
        qq = q[idx] + (q[idx + 1] - q[idx]) * f[:, None]
        cc = Cc[idx + 1] / np.repeat(k, k)[:, None] ** 0.3
        x = (qq[:, 0] / ext * 0.5 + 0.5) * size
        y = (0.5 - qq[:, 1] / ext * 0.5) * size
        ok = (x >= 0) & (x < size - 1) & (y >= 0) & (y < size - 1)
        x, y, cc = x[ok], y[ok], cc[ok]
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(img, (y0 + dy, x0 + dx), cc * w[:, None] * 0.05 * gain)
    glow = np.stack([gaussian_filter(img[..., i], 5) for i in range(3)], -1)
    out = 1 - np.exp(-(img + glow * 1.5) * 2.2)
    im = Image.fromarray((np.clip(out, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Georgia.ttf', int(size / 90))
    for i, t in enumerate(W):  # label the words where their threads start
        p0 = (to3(pos(C['resid'][0][t], 0), 0)[0] - ctr) @ rot.T
        x = (p0[0] / ext * 0.5 + 0.5) * size; y = (0.5 - p0[1] / ext * 0.5) * size
        d.text((x + 6, y - size / 90), words[t].strip(), fill=(235, 228, 214), font=font)
    im.save(f'renders/flow/{name}.png')

os.makedirs('renders/flow', exist_ok=True)
render('threads_a', 0.5, 0.35)
render('threads_b', 0.0, 0.12)
render('threads_c', 0.9, 0.9)
