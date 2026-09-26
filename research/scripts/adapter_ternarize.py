"""Ternarize a fitted adapter into Bonsai's own format ({-1,0,+1} x FP16 scale per 128 input weights) and measure the loss.
Threshold/scale per group follow TWN: delta = 0.7*mean|w|, scale = mean |w| over the kept weights. Bias is re-solved
against the ternary weights from the stored means (so the map stays centred)."""
import sys, torch, hashlib, glob
src = sys.argv[1]
A = torch.load(f'data/adapter/adapter_{src}.pt'); S = torch.load(f'data/adapter/stats_{src}.pt'); Q = torch.load('data/adapter/qwen_extra.pt')
W = A['W']                                   # [in, out] used as x @ W  ->  Linear weight is W.T [out, in]
Wl = W.T.contiguous()                        # [7680, in]
g = Wl.reshape(Wl.shape[0], -1, 128)
delta = 0.7 * g.abs().mean(-1, keepdim=True)
t = torch.sign(g) * (g.abs() > delta)
scale = (g.abs() * (t != 0)).sum(-1, keepdim=True) / (t != 0).sum(-1, keepdim=True).clamp_min(1)
Wt = (t * scale).reshape_as(Wl).T.contiguous()
n = S['n']; dx = W.shape[0]
mx = S['XtX'][:dx, dx].float() / n; my = S['XtY'][dx].float() / n
bt = my - mx @ Wt
print(f'zeros in ternary adapter: {(t == 0).float().mean():.1%}')
for name, Wx, bx in [('float', W, A['b']), ('ternary', Wt, bt)]:
    coss = []
    for k in range(S['held'].shape[0]):
        nq = Q['held_n'][k]
        yp = S['held'][k, 3:nq] @ Wx + bx; y = Q['held'][k, 3:nq]
        coss += [torch.nn.functional.cosine_similarity(yp[:, L*2560:(L+1)*2560], y[:, L*2560:(L+1)*2560], dim=-1).mean().item() for L in range(3)]
    print(f'{src} {name:8s} held-out cosine {sum(coss)/len(coss):.3f}')
params = W.numel()
print(f'params {params/1e6:.1f}M  float32 {params*4/1e6:.0f} MB  float16 {params*2/1e6:.0f} MB  ternary g128 ~{params*(1.585+16/128)/8/1e6:.1f} MB (2-bit slots {params*(2+16/128)/8/1e6:.1f} MB)')
torch.save(dict(W=Wt, b=bt), f'data/adapter/adapter_{src}t.pt')
