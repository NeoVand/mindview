"""The pyramid test: how much of the finished picture each depth of the painter already determines, at each spatial
resolution; and how coarse each block's own contribution is.

Runs Best (4 steps, the ternary painter, the turbo conditioning) on the 12 held-out prompts, reads every block's image
tokens through the tuned lens (viz.bin), decodes the lens picture with TAEF2 and compares it with the finished picture
at 512, 256 and 128 px (32 x 32, 16 x 16 and 8 x 8 tokens). Also, for every block's update dh (what the block adds to
the residual stream), the share of its energy kept by 2 x 2 and 4 x 4 average pooling: if a block's update is nearly
all coarse, the block could have run on the coarse grid.
    python scripts/pyramid_probe.py [n_prompts]
Writes renders/pyramid/metrics.json, figure.png and a contact sheet.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_1step as d1  # noqa: E402
import distill_steps as ds  # noqa: E402

D, dev, dt = d1.D, d1.dev, d1.dt
OUT = 'renders/pyramid'
N_PROMPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SIDE, NI, CIN, DIM = 32, 1024, 128, 3072
SIG = ds.SIG4  # [1.0, 0.958, 0.884, 0.717, 0.0]


def load_lens():
    viz = json.load(open('../static/models/bonsai-image-4b/viz.json'))
    e = viz['lens']
    raw = np.fromfile('../static/models/bonsai-image-4b/viz.bin', dtype=np.uint8)
    h = raw[e['offset']:e['offset'] + e['bytes']].view(np.float16).reshape(e['shape'])  # [25, 3073, 128]
    return torch.from_numpy(h.astype(np.float32)).to(dev)


def main():
    torch.manual_seed(0)
    pipe = d1.base_pipe()
    tr = d1.painter(d1.TERN()).to(dev).eval()
    dec = d1.taef2().to(dev)
    lens = load_lens()
    held = torch.load(f'{D}/held.pt', map_location='cpu', weights_only=False)
    prompts, ctxs = held['prompts'][:N_PROMPTS], held['ctx'][:N_PROMPTS]
    import lpips
    percept = lpips.LPIPS(net='vgg', verbose=False).to(dev).eval()

    # hooks: every block's image tokens after the block (double blocks return (txt, img); single blocks return the
    # concatenation [txt, img]); the image tokens are the last NI rows either way
    states = []
    def hook(mod, inp, out):
        h = out[1] if isinstance(out, tuple) else out
        states.append(h[:, -NI:].detach().float())
    hooks = [b.register_forward_hook(hook) for b in list(tr.transformer_blocks) + list(tr.single_transformer_blocks)]

    def pixels(x):  # packed latent [B, 1024, 128] -> image in [-1, 1]
        return dec(pipe._unpatchify_latents(x.permute(0, 2, 1).reshape(-1, CIN, SIDE, SIDE))).clamp(0, 1) * 2 - 1

    def down(img, k):
        return F.avg_pool2d(img, k) if k > 1 else img

    def pooled_fraction(dh, k):  # energy share of the k x k average-pooled field, per token grid
        f = dh.reshape(SIDE, SIDE, DIM).permute(2, 0, 1)[None]
        p = F.interpolate(F.avg_pool2d(f, k), scale_factor=k, mode='nearest')
        return (p.pow(2).sum() / f.pow(2).sum().clamp_min(1e-12)).item()

    depths = 4 * 25
    acc = {k: np.zeros(depths) for k in ('mse512', 'mse256', 'mse128', 'lpips512', 'lpips128', 'upd8', 'upd16',
                                          'state8', 'state16', 'updnorm')}
    sheet = []
    t0 = time.time()
    with torch.no_grad():
        for i, (prompt, ctx) in enumerate(zip(prompts, ctxs)):
            ctx = ctx[None].to(dev)
            z, _ = d1.noise([5000 + i], pipe)
            x = z.float()
            lens_x0 = []  # per depth: the lens's guess of the finished latent
            for s in range(4):
                states.clear()
                v = ds.velocity(tr, x, SIDE, SIG[s], ctx, pipe)
                x_in = x
                # the block updates and their coarseness; the lens's picture after each block
                prev = None
                for b, h in enumerate(states):
                    h1 = h[0]
                    if prev is not None:
                        dh = h1 - prev
                        d = s * 25 + b
                        acc['upd8'][d] += pooled_fraction(dh, 4)
                        acc['upd16'][d] += pooled_fraction(dh, 2)
                        acc['updnorm'][d] += (dh.norm() / prev.norm()).item()
                    acc['state8'][s * 25 + b] += pooled_fraction(h1, 4)
                    acc['state16'][s * 25 + b] += pooled_fraction(h1, 2)
                    prev = h1
                    vh = h1 @ lens[b, :DIM] + lens[b, DIM]
                    lens_x0.append(x_in[0] - SIG[s] * vh)
                x = x + (SIG[s + 1] - SIG[s]) * v  # Euler along the flow
            final = pixels(x)  # the finished picture (sigma 0)
            finals = {k: down(final, k) for k in (1, 2, 4)}
            for d, x0h in enumerate(lens_x0):
                img = pixels(x0h[None])
                for k, key in ((1, 'mse512'), (2, 'mse256'), (4, 'mse128')):
                    a, f = down(img, k), finals[k]
                    acc[key][d] += ((a - f).pow(2).mean() / f.var()).item()
                acc['lpips512'][d] += percept(img, final).item()
                acc['lpips128'][d] += percept(down(img, 4), finals[4]).item()
                if i == 0 and (d % 25 in (0, 4, 9, 14, 19, 24)):
                    sheet.append(((img[0] + 1) / 2).clamp(0, 1).cpu())
            if i == 0:
                sheet.append(((final[0] + 1) / 2).clamp(0, 1).cpu())
            print(f'{i + 1}/{len(prompts)} {time.time() - t0:.0f}s {prompt[:50]}', flush=True)
    for h in hooks:
        h.remove()
    n = len(prompts)
    m = {k: (v / n).tolist() for k, v in acc.items()}
    m['sigmas'] = SIG
    json.dump(m, open(f'{OUT}/metrics.json', 'w'))

    # the table: per step, at blocks 1, 5, 10, 15, 20, 25
    print('\nstep block   relMSE 512   256   128 | LPIPS 512  128 | update kept by 16x16  8x8 | state 16x16  8x8')
    for s in range(4):
        for b in (0, 4, 9, 14, 19, 24):
            d = s * 25 + b
            print(f'{s + 1:4} {b + 1:5}   {m["mse512"][d]:8.3f} {m["mse256"][d]:6.3f} {m["mse128"][d]:6.3f} | '
                  f'{m["lpips512"][d]:8.3f} {m["lpips128"][d]:5.3f} | {m["upd16"][d]:15.3f} {m["upd8"][d]:5.3f} | '
                  f'{m["state16"][d]:9.3f} {m["state8"][d]:5.3f}')

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = np.arange(depths) + 1
    fig, ax = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for key, lab in (('mse512', '512 px (32 x 32 tokens)'), ('mse256', '256 px (16 x 16)'), ('mse128', '128 px (8 x 8)')):
        ax[0].plot(xs, m[key], label=lab)
    ax[0].set_yscale('log'); ax[0].set_ylabel('relative MSE of the lens picture\nto the finished picture'); ax[0].legend()
    ax[1].plot(xs, m['lpips512'], label='LPIPS at 512'); ax[1].plot(xs, m['lpips128'], label='LPIPS at 128')
    ax[1].set_ylabel('LPIPS to the finished picture'); ax[1].legend()
    ax[2].plot(xs, m['upd16'], label="block's update: energy kept at 16 x 16")
    ax[2].plot(xs, m['upd8'], label="kept at 8 x 8")
    ax[2].plot(xs, m['state16'], ':', label='state: kept at 16 x 16'); ax[2].plot(xs, m['state8'], ':', label='kept at 8 x 8')
    ax[2].set_ylim(0, 1); ax[2].set_ylabel('share of energy'); ax[2].legend(); ax[2].set_xlabel('depth: step x 25 + block')
    for a in ax:
        for s in range(1, 4):
            a.axvline(s * 25 + 0.5, color='k', alpha=0.2)
    fig.suptitle(f'Best (4 steps) on {n} held-out prompts: what each depth knows, and how coarse each block\'s work is')
    fig.tight_layout(); fig.savefig(f'{OUT}/figure.png', dpi=110)

    # contact sheet: prompt 0, blocks 1/5/10/15/20/25 of each step (rows), then the final
    from PIL import Image
    tiles = [Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((192, 192)) for t in sheet]
    W = Image.new('RGB', (6 * 192, 4 * 192 + 192))
    for k, t in enumerate(tiles[:-1]):
        W.paste(t, ((k % 6) * 192, (k // 6) * 192))
    W.paste(tiles[-1].resize((192, 192)), (0, 4 * 192))
    W.save(f'{OUT}/sheet.jpg', quality=88)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
