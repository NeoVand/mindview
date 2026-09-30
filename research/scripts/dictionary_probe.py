"""Does a token consult all 12,288 keys of an MLP, or a few? For every MLP (double blocks' ff, single blocks' fused
mlp), the per-token contribution of hidden unit i to the block's output is |gelu(a_i)| * ||W_out[:, i]||. Sorted per
token, how many units carry 90 / 95 / 99% of the total contribution (L1) and of the output energy (L2, on the actual
partial sums)? One Best step at sigma 1 and one at sigma 0.717, 4 held-out prompts.
    python scripts/dictionary_probe.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_1step as d1  # noqa: E402
import distill_steps as ds  # noqa: E402

D, dev, dt = d1.D, d1.dev, d1.dt
NI, DIM, HID = 1024, 3072, 9216  # SwiGLU: 9,216 hidden units per MLP
res = {}
REC = [True]


def coverage(contrib, fracs=(0.9, 0.95, 0.99)):
    """contrib: [tokens, units] nonnegative. For each token, the smallest k whose top-k sum reaches frac of the total."""
    s, _ = contrib.sort(-1, descending=True)
    c = s.cumsum(-1) / s.sum(-1, keepdim=True).clamp_min(1e-12)
    return [((c < f).sum(-1).float().mean() + 1).item() for f in fracs]


def topk_l2(h, W, ks=(512, 1024, 2048, 4608)):
    """Relative L2 error of the MLP output when each token keeps only its top-k units (by |h_i| ||w_i||)."""
    full = h @ W  # [tokens, out]
    norms = W.norm(dim=1)  # per unit
    score = h.abs() * norms
    out = {}
    order = score.argsort(-1, descending=True)
    for k in ks:
        mask = torch.zeros_like(h)
        mask.scatter_(1, order[:, :k], 1.0)
        approx = (h * mask) @ W
        out[k] = ((approx - full).norm() / full.norm()).item()
    return out


def hook_ff(name, W_out):
    def f(mod, inp, out):
        if not REC[0]:
            return
        h = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        h = h[-NI:] if h.shape[0] > NI else h  # image rows
        h = h[:, -HID:]  # the single blocks' out-projection input is [attn_out (3072), swiglu(mlp) (9216)]
        W = W_out.detach().float()[:, -HID:].T  # [HID, out]
        contrib = h.abs() * W.norm(dim=1)[None]
        r = res.setdefault(name, dict(n=0, cov=[0.0, 0.0, 0.0], l2={}))
        cov = coverage(contrib)
        r['cov'] = [a + b for a, b in zip(r['cov'], cov)]
        for k, e in topk_l2(h, W).items():
            r['l2'][k] = r['l2'].get(k, 0.0) + e
        r['n'] += 1
    return f


def main():
    torch.manual_seed(0)
    pipe = d1.base_pipe()
    tr = d1.painter(d1.TERN()).to(dev).eval()
    held = torch.load(f'{D}/held.pt', map_location='cpu', weights_only=False)
    prompts, ctxs = held['prompts'][:4], held['ctx'][:4]
    hooks = []
    for i, b in enumerate(tr.transformer_blocks):
        hooks.append(b.ff.linear_out.register_forward_hook(hook_ff(f'double {i + 1}', b.ff.linear_out.weight)))
    for i, b in enumerate(tr.single_transformer_blocks):
        hooks.append(b.attn.to_out.register_forward_hook(hook_ff(f'single {i + 1}', b.attn.to_out.weight)))
    with torch.no_grad():
        for i, (prompt, ctx) in enumerate(zip(prompts, ctxs)):
            ctx = ctx[None].to(dev)
            z, _ = d1.noise([5000 + i], pipe)
            x = z.float()
            for s in range(4):
                REC[0] = s in (0, 3)
                v = ds.velocity(tr, x, 32, ds.SIG4[s], ctx, pipe)
                x = x + (ds.SIG4[s + 1] - ds.SIG4[s]) * v
            print(f'{i + 1}/4 done', flush=True)
    for h in hooks:
        h.remove()
    print(f"{'MLP':12} {'units for 90%':>13} {'95%':>7} {'99%':>7} | rel L2 error keeping top-k of 9,216 per token: "
          f"{'512':>6} {'1024':>6} {'2048':>6} {'4608':>6}")
    for name, r in res.items():
        n = r['n']
        cov = [c / n for c in r['cov']]
        l2 = {k: v / n for k, v in r['l2'].items()}
        print(f"{name:12} {cov[0]:13.0f} {cov[1]:7.0f} {cov[2]:7.0f} | {'':46} {l2[512]:6.3f} {l2[1024]:6.3f} {l2[2048]:6.3f} {l2[4608]:6.3f}")


if __name__ == '__main__':
    main()
