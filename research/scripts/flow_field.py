"""Is the network a flow we can see? Sample the real vector field of a Bonsai 1.7B layer on a plane through meaning-space.

Each layer does h <- h + attn(h) + mlp(h + attn(h)): one Euler step of a vector field. Here the MLP half of that field
is evaluated at a grid of probe points on a 2D plane through the followed word's position, using the real ternary
weights, and drawn as streamlines. The gate neurons are drawn as the lines where their hyperplanes cut the plane
(membranes: a neuron switches on as a point crosses its line).

usage: flow_field.py [layers=3,8,14,20,26] [scale=0.3]
writes renders/flow/field_L{l}_s{scale}.png
"""
import sys, glob, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from transformers import AutoTokenizer, AutoModelForCausalLM

layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '3,8,14,20,26').split(',')]
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
prompt = os.environ.get('PROMPT', 'a bonsai tree made of glowing circuitry in a dark museum, volumetric light')
RES = 128
path = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-unpacked/snapshots/*/')[0]
dev = 'mps'
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, attn_implementation='eager').to(dev).eval()
text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors='pt').input_ids.to(dev)
words = [tok.decode([i]) for i in ids[0]]
end = next(i for i, w in enumerate(words) if i > 3 and w == '<|im_end|>')
t = end - 1
print('following', repr(words[t]), 'token', t, 'of', len(words))

resid, attn_out = {}, {}
hooks = []
for i, L in enumerate(model.model.layers):
    hooks.append(L.register_forward_pre_hook(lambda m, a, i=i: resid.__setitem__(i, a[0][0].float().cpu()), with_kwargs=False))
    hooks.append(L.self_attn.register_forward_hook(lambda m, a, o, i=i: attn_out.__setitem__(i, o[0][0].float().cpu())))
with torch.no_grad():
    out = model(input_ids=ids, output_hidden_states=True)
for h in hooks:
    h.remove()

os.makedirs('renders/flow', exist_ok=True)
torch.set_grad_enabled(False)
for l in layers:
    L = model.model.layers[l]
    h = resid[l]
    mid = h + attn_out[l]
    c = mid[t].double()
    with torch.no_grad():
        m_true = L.mlp(L.post_attention_layernorm(mid[t:t + 1].to(dev).half()))[0].float().cpu().double()
    a = attn_out[l][t].double()
    e1 = m_true / m_true.norm()
    e2 = a - (a @ e1) * e1
    e2 = e2 / e2.norm()
    R = scale * c.norm().item()
    g = torch.linspace(-R, R, RES, dtype=torch.float64)
    A, B = torch.meshgrid(g, g, indexing='xy')
    P = c[None] + A.reshape(-1, 1) * e1[None] + B.reshape(-1, 1) * e2[None]
    F = []
    with torch.no_grad():
        for chunk in P.split(4096):
            x = chunk.float().half().to(dev)
            F.append(L.mlp(L.post_attention_layernorm(x)).float().cpu().double())
    F = torch.cat(F)
    fx, fy = (F @ e1).reshape(RES, RES), (F @ e2).reshape(RES, RES)
    off = (F - (F @ e1)[:, None] * e1[None] - (F @ e2)[:, None] * e2[None]).norm(dim=1).reshape(RES, RES)
    mag = torch.sqrt(fx ** 2 + fy ** 2)

    # membranes: gate neuron j is zero where (w_j * g) . p = 0 (RMSNorm only rescales p)
    Wg = L.mlp.gate_proj.weight.float().cpu().double() * L.post_attention_layernorm.weight.float().cpu().double()[None]
    k0, k1, k2 = Wg @ c, Wg @ e1, Wg @ e2
    gate_here = (Wg @ c).numpy()
    segs, cols = [], []
    for j in range(Wg.shape[0]):
        n1, n2, d = k1[j].item(), k2[j].item(), k0[j].item()
        # line: d + n1*x + n2*y = 0, clip to the square
        if abs(n2) > abs(n1):
            xs = np.array([-R, R]); ys = -(d + n1 * xs) / n2
        else:
            ys = np.array([-R, R]); xs = -(d + n2 * ys) / n1
        if np.all(np.abs(xs) > R * 1.5) or np.all(np.abs(ys) > R * 1.5):
            continue
        segs.append(np.stack([xs, ys], 1))
        cols.append((1.0, 0.55, 0.2, 0.05) if gate_here[j] > 0 else (0.3, 0.7, 1.0, 0.05))
    print(f'layer {l}: |c|={c.norm():.1f} |mlp|={m_true.norm():.2f} |attn|={a.norm():.2f} R={R:.1f} membranes in view {len(segs)} '
          f'field |in-plane| {mag.mean():.2f} |off| {off.mean():.2f}')

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    fig.patch.set_facecolor('black'); ax.set_facecolor('black')
    ax.imshow(np.log1p(mag.numpy()), extent=[-R, R, -R, R], origin='lower', cmap='bone', alpha=0.55)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.4))
    ax.streamplot(g.numpy(), g.numpy(), fx.numpy(), fy.numpy(), density=2.4, color=np.log1p(mag.numpy()), cmap='copper', linewidth=0.7, arrowsize=0.6)
    # the word itself and where the MLP sends it
    ax.plot([0], [0], 'o', color='white', ms=5)
    ax.arrow(0, 0, m_true @ e1, m_true @ e2, color='white', width=R * 0.004, length_includes_head=True)
    # other words projected onto the plane (they are mostly off it)
    for s in range(len(words)):
        d = mid[s].double() - c
        x, y = (d @ e1).item(), (d @ e2).item()
        if abs(x) < R and abs(y) < R and s != t:
            ax.text(x, y, words[s].strip() or '·', color=(0.9, 0.88, 0.8, 0.6), fontsize=7)
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.axis('off')
    ax.set_title(f'layer {l + 1}: the MLP field around "{words[t].strip()}"', color='white')
    fig.savefig(f'renders/flow/field_L{l + 1}_s{scale}.png', bbox_inches='tight', facecolor='black')
    plt.close(fig)
