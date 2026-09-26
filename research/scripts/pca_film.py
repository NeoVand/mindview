"""Turn captured per-block PCA fields into a smooth film: sign-align components block-to-block so colours
don't flicker, upsample, and crossfade. Rows of the sheet = steps, 25 blocks each."""
import sys, os, subprocess
import numpy as np
from PIL import Image
d = sys.argv[1]
pca = np.load(f'{d}/capture.npz')['pca']  # [steps*25, N, 3]
n, N, _ = pca.shape
hw = int(N ** 0.5)
aligned = [pca[0]]
for i in range(1, n):
    p = pca[i].copy()
    for c in range(3):  # flip component sign to best match the previous frame
        if (p[:, c] * aligned[-1][:, c]).sum() < 0: p[:, c] *= -1
    aligned.append(p)
os.makedirs(f'{d}/pca_frames', exist_ok=True)
k = 0
prev = None
for i, p in enumerate(aligned):
    lo, hi = np.percentile(p, 1, 0), np.percentile(p, 99, 0)  # per-frame: block norms grow ~80x with depth
    q = np.clip((p - lo) / (hi - lo + 1e-6), 0, 1).reshape(hw, hw, 3)
    im = Image.fromarray((q * 255).astype(np.uint8)).resize((768, 768), Image.BICUBIC)
    for t in np.linspace(0, 1, 6, endpoint=False):  # 6-frame crossfade per block
        fr = im if prev is None else Image.blend(prev, im, float(t))
        fr.save(f'{d}/pca_frames/{k:05d}.png'); k += 1
    prev = im
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-i', f'{d}/pca_frames/%05d.png', '-pix_fmt', 'yuv420p', '-c:v', 'libx264', '-crf', '18', f'{d}/pca_film.mp4'])
print('frames', k, '->', f'{d}/pca_film.mp4')
