"""What the painter's activations look like at the inputs of its big matrices (for integer / low-bit paths):
per-token int8 and int4 quantization error (cosine to bf16), outliers, and how sparse the MLP's GELU output is.
One Best step (sigma 1) at 32 x 32 on a few prompts, hooks on every Linear of the single blocks and double blocks.
    python scripts/act_stats.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_1step as d1  # noqa: E402

D, dev, dt = d1.D, d1.dev, d1.dt
PROMPTS = ['a red fox sleeping in the snow at dawn', 'a neon sign that says OPEN in a rainy window',
           'a cat wearing a tiny astronaut helmet, studio photo', 'three apples and one pear on a wooden table']


def qcos(x, bits):
    """Cosine between x and its per-row symmetric quantization to `bits` (absmax scaling)."""
    q = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / q
    xq = (x / s).round().clamp(-q, q) * s
    return torch.nn.functional.cosine_similarity(x, xq, dim=-1)


def qcos_clip(x, bits, keep=0.999):
    """The same, clipping each row at its `keep` quantile (outliers saturate)."""
    q = 2 ** (bits - 1) - 1
    s = x.abs().float().quantile(keep, dim=-1, keepdim=True).clamp_min(1e-8) / q
    xq = (x / s).round().clamp(-q, q) * s
    return torch.nn.functional.cosine_similarity(x, xq, dim=-1)


stats = {}


def hook(name):
    def f(mod, inp, out):
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        if 'img' in name or 'single' in name:
            x = x[-1024:] if x.shape[0] > 1024 else x  # the picture's rows (the text rows come first)
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        st = stats.setdefault(name, dict(n=0, c8=0.0, c4=0.0, c4c=0.0, c6=0.0, out8=0.0, zero=0.0))
        st['n'] += 1
        st['c8'] += qcos(x, 8).mean().item()
        st['c6'] += qcos(x, 6).mean().item()
        st['c4'] += qcos(x, 4).mean().item()
        st['c4c'] += qcos_clip(x, 4).mean().item()
        st['out8'] += (x.abs() > 8 * rms).float().sum(-1).mean().item()  # outliers per row
        st['zero'] += (x.abs() < 0.02 * rms).float().mean().item()  # near-zero share (sparsity)
    return f


def main():
    torch.manual_seed(0)
    pipe = d1.base_pipe()
    tr = d1.painter(d1.TERN()).to(dev).eval()
    ctx = d1.conditioning(PROMPTS)
    z, ids = d1.noise([7] * len(PROMPTS), pipe)
    hooks = []
    for n, m in tr.named_modules():
        if isinstance(m, nn.Linear) and ('single_transformer_blocks' in n or 'transformer_blocks' in n):
            hooks.append(m.register_forward_hook(hook(n)))
    with torch.no_grad():
        d1.x0(tr, z, ids, ctx, pipe)
    for h in hooks:
        h.remove()
    # summarise by kind of matrix
    kinds = {}
    for n, st in stats.items():
        kind = n.split('.')[-1] if 'single' not in n else 'single.' + n.split('.')[-1]
        if 'attn' in n and 'single' not in n:
            kind = ('img.' if 'add_' not in n and 'to_' in n else 'txt.') + n.split('.')[-1]
        k = kinds.setdefault(kind, dict(n=0, c8=0.0, c6=0.0, c4=0.0, c4c=0.0, out8=0.0, zero=0.0))
        for key in ('c8', 'c6', 'c4', 'c4c', 'out8', 'zero'):
            k[key] += st[key] / st['n']
        k['n'] += 1
    print(f"{'matrix (input)':34} {'cos int8':>8} {'int6':>6} {'int4':>6} {'int4 clip':>9} {'outliers/row':>12} {'near-zero':>9}")
    for kind, k in sorted(kinds.items()):
        n = k['n']
        print(f"{kind:34} {k['c8'] / n:8.4f} {k['c6'] / n:6.4f} {k['c4'] / n:6.4f} {k['c4c'] / n:9.4f} {k['out8'] / n:12.1f} {k['zero'] / n:9.3f}")


if __name__ == '__main__':
    main()
