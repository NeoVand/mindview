"""Concept stills for drawing the painter's computation as threads (data from painter_space.py).

Every other patch (16 x 16 of 32 x 32) followed through 4 steps x 25 blocks, projected on the fixed directions, each
coloured by the colour its patch has in the finished picture. Left: side view (x = block, y = first direction).
Right: cross-sections (first vs second direction) at a few blocks. Also the prompt's words.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

d = torch.load('data/painter_space.pt')
E, Et = d['E_img'].float(), d['E_txt'].float()
img = d['img'].float()                     # [100, 256, 3072] centred / scaled per block
txt = d['txt'].float()                     # [100, words, 3072]
P = torch.einsum('bnd,kd->bnk', img, E).numpy()          # [100, 256, 3]
T = torch.einsum('bnd,kd->bnk', txt, Et).numpy()
pic = d['picture']                                          # [3, 512, 512]
col = F.avg_pool2d(pic[None], 32)[0].reshape(3, -1).T.numpy()   # 16 x 16 patch colours
col = np.clip(col, 0, 1)
print('projection ranges', P.min(axis=(0, 1)), P.max(axis=(0, 1)))

fig = plt.figure(figsize=(22, 13), facecolor='black')
for k, (a, b) in enumerate([(0, 0), (1, 0)]):
    ax = fig.add_axes([0.03, 0.52 - 0.47 * k, 0.62, 0.44], facecolor='black')
    for n in range(P.shape[1]):
        ax.plot(np.arange(P.shape[0]), P[:, n, k], color=col[n], lw=0.5, alpha=0.8)
    for s in range(1, 4):
        ax.axvline(25 * s - 0.5, color='#333', lw=0.5)
    ax.set_title(f'patches, direction {k + 1} (x: block, 4 passes of 25)', color='w')
    ax.axis('off')
for j, blk in enumerate([0, 12, 24, 49, 74, 99]):
    ax = fig.add_axes([0.67 + 0.11 * (j % 3), 0.55 - 0.45 * (j // 3), 0.1, 0.4], facecolor='black')
    ax.scatter(P[blk, :, 0], P[blk, :, 1], c=col, s=6)
    ax.set_title(f'block {blk % 25 + 1}, pass {blk // 25 + 1}', color='w', fontsize=9)
    ax.axis('off')
fig.savefig('renders/concepts/painter_space_patches.png', dpi=80)

fig = plt.figure(figsize=(22, 6), facecolor='black')
ax = fig.add_axes([0.02, 0.05, 0.96, 0.85], facecolor='black')
cm = plt.get_cmap('coolwarm')
for n in range(T.shape[1]):
    ax.plot(np.arange(T.shape[0]), T[:, n, 0], color=cm(n / max(1, T.shape[1] - 1)), lw=1)
ax.set_title('the prompt\'s tokens in the painter, direction 1', color='w')
ax.axis('off')
fig.savefig('renders/concepts/painter_space_words.png', dpi=80)
print('saved')
