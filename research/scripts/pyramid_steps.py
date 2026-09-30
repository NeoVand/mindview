"""Coarse steps, not coarse blocks: is Best's quality reachable when the first steps run at 16 x 16 tokens?
The ternary painter (turbo conditioning, no branches) on the 12 held-out prompts:
  A  4 steps at 32 x 32 (Best)
  B  3 steps at 16 x 16, the guess enlarged and noised back to sigma_3, 1 step at 32 x 32   (44% of A's work)
  C  2 steps at 16 x 16, 2 at 32 x 32                                                       (62%)
Reports LPIPS of B and C to A (difference, not quality) and writes a contact sheet to judge quality by eye.
    python scripts/pyramid_steps.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_1step as d1  # noqa: E402
import distill_steps as ds  # noqa: E402

D, dev = d1.D, d1.dev
SIG = ds.SIG4
OUT = 'renders/pyramid'


def main():
    torch.manual_seed(0)
    pipe = d1.base_pipe()
    tr = d1.painter(d1.TERN()).to(dev).eval()
    dec = d1.taef2().to(dev)
    import lpips
    percept = lpips.LPIPS(net='vgg', verbose=False).to(dev).eval()
    held = torch.load(f'{D}/held.pt', map_location='cpu', weights_only=False)

    def pixels(x, side):
        return dec(pipe._unpatchify_latents(x.permute(0, 2, 1).reshape(-1, 128, side, side))).clamp(0, 1) * 2 - 1

    def run(ctx, seed, coarse):
        """coarse: how many of the 4 steps run at 16 x 16 (the rest at 32 x 32)."""
        g = torch.Generator('cpu').manual_seed(seed)
        x = ds.noise([seed], 16 if coarse else 32, pipe).float()
        for s in range(4):
            if s == coarse and coarse > 0:
                x0 = x - SIG[s] * v  # the last coarse step's guess of the clean picture
                up = ds.upsample(x0, pipe)
                eps = ds.noise([seed + 1], 32, pipe).float()
                x = (1 - SIG[s]) * up + SIG[s] * eps  # noised back to sigma_s at full size
            side = 16 if s < coarse else 32
            v = ds.velocity(tr, x, side, SIG[s], ctx, pipe)
            x = x + (SIG[s + 1] - SIG[s]) * v
        return x

    from PIL import Image
    import numpy as np
    rows = []
    scores = {'B': [], 'C': []}
    t0 = time.time()
    with torch.no_grad():
        for i, (prompt, ctx) in enumerate(zip(held['prompts'], held['ctx'])):
            ctx = ctx[None].to(dev)
            seed = 5000 + i
            A = pixels(run(ctx, seed, 0), 32)
            B = pixels(run(ctx, seed, 3), 32)
            C = pixels(run(ctx, seed, 2), 32)
            scores['B'].append(percept(B, A).item())
            scores['C'].append(percept(C, A).item())
            rows.append([((t[0] + 1) / 2).clamp(0, 1).cpu() for t in (A, C, B)])
            print(f'{i + 1}/12 {time.time() - t0:.0f}s  LPIPS to Best: 2 coarse {scores["C"][-1]:.3f}, 3 coarse {scores["B"][-1]:.3f}  {prompt[:40]}', flush=True)
    print(f'mean LPIPS to Best: 2 coarse steps {np.mean(scores["C"]):.3f}, 3 coarse steps {np.mean(scores["B"]):.3f}')
    W = Image.new('RGB', (3 * 256, len(rows) * 256))
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            W.paste(Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((256, 256)), (c * 256, r * 256))
    W.save(f'{OUT}/steps.jpg', quality=88)
    print('wrote', OUT, '(columns: Best | 2 coarse + 2 fine | 3 coarse + 1 fine)')


if __name__ == '__main__':
    main()
