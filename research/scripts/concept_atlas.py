"""Concept: an atlas of meaning. A plane through the prompt's words at one layer, drawn as a map.

- Regions: every point of the plane is read out by the model's own vocabulary (final norm + lm_head, argmax): the
  region labelled "lamp" is where the row for "lamp" has the highest dot product with the point.
- Contour lines: the MLP's gate neurons. Each is a hyperplane; where it cuts the plane a point switches that neuron on.
- Routes: attention between the words at this layer, drawn like flight routes.
usage: concept_atlas.py [layers=6,12,18,24]
"""
import sys, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import ndimage
from flow_common import load, capture, logit_lens

plt.rcParams['font.family'] = ['Georgia', 'Songti SC']
layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '6,12,18,24').split(',')]
RES = 220
tok, model = load()
C = capture(tok, model)
words = C['words']
W = list(range(3, C['end']))  # the prompt's own words
os.makedirs('renders/flow', exist_ok=True)

for l in layers:
    X = C['mid'][l][W].double()
    mu = X.mean(0)
    U, S, Vt = torch.linalg.svd(X - mu, full_matrices=False)
    e = Vt[:2]  # 2 x D
    xy = ((X - mu) @ e.T).numpy()
    lo, hi = xy.min(0), xy.max(0)
    span = (hi - lo).max() * 0.62
    cx, cy = (lo + hi) / 2
    gx = np.linspace(cx - span, cx + span, RES)
    gy = np.linspace(cy - span, cy + span, RES)
    A, B = np.meshgrid(gx, gy)
    P = mu[None] + torch.from_numpy(A.reshape(-1, 1)) * e[0][None] + torch.from_numpy(B.reshape(-1, 1)) * e[1][None]
    ids, conf = logit_lens(model, P)
    ids = ids.numpy().reshape(RES, RES)
    conf = conf.numpy().reshape(RES, RES)
    own, _ = logit_lens(model, C['mid'][l][W])

    # colour each region by a stable hue from its token id
    uniq = np.unique(ids)
    rng = np.random.default_rng(0)
    hue = {u: rng.random() for u in uniq}
    img = np.zeros((RES, RES, 3))
    for u in uniq:
        m = ids == u
        h = hue[u]
        col = np.array(matplotlib.colors.hsv_to_rgb([h, 0.45, 1.0]))
        img[m] = col
    img *= (0.05 + 0.18 * conf[..., None])
    edge = (np.diff(ids, axis=0, prepend=ids[:1]) != 0) | (np.diff(ids, axis=1, prepend=ids[:, :1]) != 0)
    img[edge] = [0.55, 0.52, 0.46]

    fig, ax = plt.subplots(figsize=(12, 12), dpi=120)
    fig.patch.set_facecolor('black'); ax.set_facecolor('black')
    ax.imshow(img, extent=[gx[0], gx[-1], gy[0], gy[-1]], origin='lower', interpolation='nearest')

    # gate neuron membranes (RMSNorm only rescales, so a gate is zero on a hyperplane through the origin)
    L = model.model.layers[l]
    Wg = L.mlp.gate_proj.weight.float().cpu().double() * L.post_attention_layernorm.weight.float().cpu().double()[None]
    k0, k1, k2 = (Wg @ mu).numpy(), (Wg @ e[0]).numpy(), (Wg @ e[1]).numpy()
    segs = []
    for d, n1, n2 in zip(k0, k1, k2):
        if abs(n2) > abs(n1):
            xs = np.array([gx[0], gx[-1]]); ys = -(d + n1 * xs) / n2
        else:
            ys = np.array([gy[0], gy[-1]]); xs = -(d + n2 * ys) / n1
        segs.append(np.stack([xs, ys], 1))
    ax.add_collection(LineCollection(segs, colors=[(0.9, 0.86, 0.78, 0.035)], linewidths=0.35))

    # region names
    for u in uniq:
        lab, n = ndimage.label(ids == u)
        for k in range(1, n + 1):
            m = lab == k
            area = m.sum()
            if area < 250:
                continue
            yy, xx = np.nonzero(m)
            name = tok.decode([int(u)]).strip() or repr(tok.decode([int(u)]))
            fs = min(22, 5 + np.sqrt(area) * 0.18)
            ax.text(gx[int(xx.mean())], gy[int(yy.mean())], name, color=(0.85, 0.8, 0.7, 0.55), fontsize=fs,
                    style='italic', ha='center', va='center')

    # attention routes between the words (all heads, the sink excluded)
    P_ = C['probs'][l].mean(0).numpy()
    routes, rc = [], []
    for i, t in enumerate(W):
        for j, s in enumerate(W):
            if s >= t:
                continue
            w = P_[t, s]
            if w < 0.02:
                continue
            a, b = xy[i], xy[j]
            mid = (a + b) / 2 + np.array([-(b - a)[1], (b - a)[0]]) * 0.25
            tt = np.linspace(0, 1, 24)[:, None]
            curve = (1 - tt) ** 2 * a + 2 * (1 - tt) * tt * mid + tt ** 2 * b
            routes.append(curve)
            rc.append((1.0, 0.62, 0.25, min(0.9, w * 3)))
    ax.add_collection(LineCollection(routes, colors=rc, linewidths=1.0))

    for i, t in enumerate(W):
        ax.plot(*xy[i], 'o', color=(1, 0.95, 0.85), ms=4)
        ax.text(xy[i][0], xy[i][1] + span * 0.018, words[t].strip(), color='white', fontsize=11, ha='center')
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1]); ax.axis('off')
    ax.set_title(f'layer {l + 1}: the plane through your words, named by the model ({len(uniq)} regions)', color='white')
    fig.savefig(f'renders/flow/atlas_L{l + 1}.png', bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f'layer {l + 1}: {len(uniq)} regions; the words read as', [tok.decode([int(i)]) for i in own])
