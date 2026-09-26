"""A fixed 'space' for drawing the painter's own computation as threads, the way the reader's words are drawn.

The painter updates a 3,072-number state per image patch (1,024 of them) and per text row (512) in each of its 25
blocks, at each step. To draw those states as threads the live piece needs two directions to project them on, and
they must be fixed before a painting starts (so the picture of block 1 does not move when block 20 arrives). This
fits them from real runs: the reference painter (painter_reference.py, exact against diffusers) runs several prompts
encoded by our own 1.7B + adapter, every block's states are recorded, each block's image rows are centred and scaled
(by the block's own mean and median norm, which the browser can compute live), and the top principal directions of all
of that are kept. Text rows (the prompt's real tokens) get their own directions, same treatment.

Writes data/painter_space.pt (directions, and one prompt's full trajectories for concept stills) and appends
'space_img' / 'space_txt' ([3, 3072] f32 each) to static/models/bonsai-image-4b/viz.bin / viz.json.
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(__file__), 'painter_reference.py')).read()
exec(compile(src[: src.index('# ================================================================== run')], 'painter_reference', 'exec'))

OUT = '../static/models/bonsai-image-4b'
PROMPTS = 10
SHOW = 'a tiny glass robot holding a daisy'   # its trajectories are kept for the concept stills
cache = []
for f in sorted(glob.glob('data/prompt_cache/*.pt')):
    d = torch.load(f, map_location='cpu')
    if d['encoder'] == 'bonsai17t_adapt':
        cache.append((f, d['prompt'], d['n_real']))
show = [c for c in cache if c[1].startswith(SHOW)]
rest = [c for c in cache if not c[1].startswith(SHOW)]
rng = np.random.default_rng(3)
chosen = show + [rest[i] for i in rng.choice(len(rest), PROMPTS - len(show), replace=False)]
print('prompts:', [c[1][:40] for c in chosen])

cov_img = torch.zeros(D, D, dtype=torch.float64)
cov_txt = torch.zeros(D, D, dtype=torch.float64)
n_img = n_txt = 0
keep = None
t0 = time.time()
with torch.no_grad():
    for pi, (f, prompt, n_real) in enumerate(chosen):
        ctx = torch.load(f, map_location='cpu')['embeds'][0].float().to(dev)
        g = torch.Generator().manual_seed(100 + pi)
        lat = torch.randn(1024, 128, generator=g).to(dev)
        traj_img, traj_txt, sig = [], [], []
        for s in range(M['schedule']['steps']):
            rec = []
            v = dit(lat, ctx, s, rec)
            for b, h in enumerate(rec[1:]):                               # after each of the 25 blocks
                x = h[N_TXT:]
                mu = x.mean(0)
                sc = (x - mu).norm(dim=1).median()
                xc = (x - mu) / sc                                        # (MPS has no float64: products in f32)
                cov_img += (xc.T @ xc).cpu().double()
                n_img += len(xc)
                t = h[3:n_real]
                mt = t.mean(0)
                st = (t - mt).norm(dim=1).median()
                tc = (t - mt) / st
                cov_txt += (tc.T @ tc).cpu().double()
                n_txt += len(tc)
                if pi == 0:  # every other patch in each direction: 16 x 16 of the 32 x 32
                    traj_img.append(xc.view(32, 32, D)[::2, ::2].reshape(256, D).cpu().half())
                    traj_txt.append(tc.cpu().half())
            lat = lat + (SIG[s + 1] - SIG[s]).item() * v
            sig.append(SIG[s].item())
        if pi == 0:
            img = taef2(lat)
            keep = dict(prompt=prompt, n_real=n_real, img=torch.stack(traj_img), txt=torch.stack(traj_txt),
                        picture=img.cpu())
        print(f'{pi + 1}/{len(chosen)} {prompt[:50]!r} {time.time() - t0:.0f}s', flush=True)


def top(cov, n, k=3):
    w, V = torch.linalg.eigh(cov / n)
    w, V = w.flip(0), V.flip(1)
    print('  explained (top 6):', [round(x, 4) for x in (w[:6] / w.sum()).tolist()])
    return V[:, :k].T.float().contiguous()


print('image rows:')
E_img = top(cov_img, n_img)
print('text rows:')
E_txt = top(cov_txt, n_txt)
torch.save(dict(E_img=E_img, E_txt=E_txt, **keep), 'data/painter_space.pt')

meta = json.load(open(f'{OUT}/viz.json'))
raw = open(f'{OUT}/viz.bin', 'rb').read()
last = max((v['offset'] + v['bytes'] for k, v in meta.items() if k not in ('space_img', 'space_txt')))
blob = raw[:last]
for name, E in (('space_img', E_img), ('space_txt', E_txt)):
    a = E.numpy().astype('<f4')
    meta[name] = dict(dtype='f32', shape=list(a.shape), offset=len(blob), bytes=a.nbytes,
                      note='principal directions of the painter states after each block (rows centred and scaled per block)')
    blob += a.tobytes()
open(f'{OUT}/viz.bin', 'wb').write(blob)
json.dump(meta, open(f'{OUT}/viz.json', 'w'), indent=1)
print('wrote', f'{OUT}/viz.bin', len(blob) / 1e6, 'MB')
