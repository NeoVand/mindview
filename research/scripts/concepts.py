"""Concept studies (stills, real data) for showing the computation causally.

A  meaning assembling: token states drift into phrases as they read each other (attention links, content only)
B  one neuron's tally: a ternary matmul as thousands of +x / -x / skip votes summing to one output
C  every block paints on itself: per-block tuned-lens readouts, and what each block changed
"""
import glob, json, sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT = 'renders/concepts'
import os
os.makedirs(OUT, exist_ok=True)
PROMPT = 'a bonsai tree made of glowing circuitry in a dark museum, volumetric light'
BG = (0, 0, 0)
MINUS = np.array([86, 200, 255]); PLUS = np.array([255, 176, 74]); BONE = (232, 226, 214); ASH = (110, 116, 124)
try:
    FONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Georgia Italic.ttf', 22)
    SMALL = ImageFont.truetype('/System/Library/Fonts/Supplemental/Georgia.ttf', 16)
except Exception:
    FONT = SMALL = ImageFont.load_default()

path = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-unpacked/snapshots/*/')[0]
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation='eager').eval()
text = tok.apply_chat_template([{'role': 'user', 'content': PROMPT}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors='pt').input_ids
toks = [tok.decode([i]) for i in ids[0].tolist()]
n = len(toks)
cap = {}
L_TALLY = 14
model.model.layers[L_TALLY].mlp.down_proj.register_forward_hook(lambda m, a, o: cap.update(x=a[0][0].detach(), y=o[0].detach()))
with torch.no_grad():
    out = model(ids, output_hidden_states=True, output_attentions=True)
H = torch.stack(out.hidden_states)[:, 0].numpy()          # [29, n, 2048]
A = torch.stack(out.attentions)[:, 0].numpy()             # [28, 16, n, n]
content = [i for i, t in enumerate(toks) if t.strip() and not t.startswith('<') and t.strip() not in ('user', 'assistant', ',')]
SINKS = [0, 1, 2]  # '<|im_start|>', 'user', '\n': where attention rests when it has nothing to read

# ---------------------------------------------------------------- A: meaning assembling
def embed2d(X):
    X = X - X.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)   # direction only: what, not how loud
    G = X @ X.T
    J = np.eye(len(X)) - 1 / len(X)
    B = -0.5 * J @ (2 - 2 * G) @ J                               # classical MDS on cosine distance
    w, V = np.linalg.eigh(B)
    return V[:, -2:][:, ::-1] * np.sqrt(np.maximum(w[-2:][::-1], 1e-9))

layers = [0, 2, 5, 9, 16, 28]
W, Hh = 2400, 1500
img = Image.new('RGB', (W, Hh), BG); d = ImageDraw.Draw(img)
prev = None
cw, ch = W // 3, Hh // 2
for k, L in enumerate(layers):
    P = embed2d(H[L][content])
    if prev is not None:                                          # Procrustes-align to the previous panel so motion is real
        U, _, Vt = np.linalg.svd(P.T @ prev); P = P @ (U @ Vt)
    prev = P
    ox, oy = (k % 3) * cw, (k // 3) * ch
    P = (P - P.mean(0)) / (np.abs(P).max() + 1e-6)
    pts = [(ox + cw / 2 + p[0] * cw * 0.36, oy + ch / 2 + p[1] * ch * 0.36) for p in P]
    if L > 0:
        att = A[L - 1].copy()                                      # the attention that produced this layer
        att[:, :, SINKS] = 0
        strength = att.max(0)                                      # strongest head for each (reader, source)
        for a_i, i in enumerate(content):
            for b_i, j in enumerate(content):
                s = strength[i, j]
                if j < i and s > 0.12:
                    col = tuple(int(c) for c in (PLUS * min(1, s * 1.6)))
                    d.line([pts[a_i], pts[b_i]], fill=col, width=max(1, int(s * 6)))
    for a_i, i in enumerate(content):
        x, y = pts[a_i]
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=tuple(int(c) for c in MINUS))
        d.text((x + 8, y - 11), toks[i].strip(), fill=BONE, font=SMALL)
    d.text((ox + 24, oy + 18), 'embeddings' if L == 0 else f'after layer {L}', fill=ASH, font=FONT)
img.save(f'{OUT}/A_meaning_assembling.png')
print('A written')

# ---------------------------------------------------------------- B: one neuron's tally
wt = model.model.layers[L_TALLY].mlp.down_proj.weight.detach().numpy()   # [2048, 6144]
g = wt.reshape(2048, -1, 128); s = np.abs(g).max(-1, keepdims=True); s[s == 0] = 1
trit = np.rint(g / s).reshape(2048, 6144); scale = np.repeat(s[..., 0], 128, axis=1)
tpos = toks.index(' tree')
x = cap['x'][tpos].numpy(); y = cap['y'][tpos].numpy()
contrib = trit * scale * x[None, :]                                    # every single vote
W, Hh = 2400, 1500
img = Image.new('RGB', (W, Hh), BG); d = ImageDraw.Draw(img)
# overview: all 12.6M votes, 2048 rows x 6144 cols, compressed 4x4 -> signed sum
ov = contrib[:, :6144].reshape(512, 4, 1536, 4).sum((1, 3))
m = np.quantile(np.abs(ov), 0.995)
v = np.clip(ov / m, -1, 1)
rgb = np.where(v[..., None] > 0, PLUS * v[..., None], MINUS * (-v[..., None])) ** 1.0
rgb = (np.abs(rgb) ** 0.8 * 255 ** 0.2).clip(0, 255).astype(np.uint8)
ovimg = Image.fromarray(rgb).resize((1536, 512), Image.NEAREST)
img.paste(ovimg, (60, 90))
d.text((60, 40), f'Layer {L_TALLY}, reading " tree": all 12.6 million ternary votes of one matrix (2048 outputs × 6144 inputs, +x amber, −x cyan, skip black)', fill=ASH, font=SMALL)
# the outputs they add up to
yy = y / (np.abs(y).max() + 1e-6)
for r in range(512):
    val = yy[r * 4:(r + 1) * 4].mean()
    col = tuple(int(c) for c in (PLUS if val > 0 else MINUS) * min(1, abs(val) * 3))
    d.line([(1610, 90 + r), (1610 + int(abs(val) * 300), 90 + r)], fill=col)
d.text((1610, 60), 'the outputs they sum to', fill=ASH, font=SMALL)
# three neurons: running sums as the 6144 votes are added one by one
top = np.argsort(-np.abs(y))[:3]
for k, i in enumerate(top):
    c = contrib[i]
    run = np.cumsum(c)
    ox, oy, w_, h_ = 60, 680 + k * 270, 2280, 220
    lim = max(np.abs(run).max(), 1e-6)
    # each vote as a tick
    for j in range(0, 6144, 2):
        t = trit[i, j]
        if t == 0: continue
        col = tuple(int(q) for q in (PLUS if c[j] > 0 else MINUS) * min(1, abs(c[j]) / (np.abs(c).max() + 1e-9) * 3 + 0.08))
        xpix = ox + j / 6144 * w_
        d.line([(xpix, oy + h_ - 14), (xpix, oy + h_ - 2)], fill=col)
    pts = [(ox + j / 6144 * w_, oy + (h_ - 30) / 2 - run[j] / lim * (h_ - 40) / 2) for j in range(0, 6144, 4)]
    d.line(pts, fill=BONE, width=2)
    d.text((ox, oy - 4), f'output neuron {i}: {int((trit[i] != 0).sum())} votes, {int((trit[i] == 0).sum())} skipped, total {y[i]:+.2f}', fill=ASH, font=SMALL)
img.save(f'{OUT}/B_one_neuron_tally.png')
print('B written')

# ---------------------------------------------------------------- C: every block paints on itself
trace = 'traces/bonsai-museum-ternary'
blocks = 25
W, Hh = 25 * 96, 3 * 96 + 60
img = Image.new('RGB', (W, Hh), BG); d = ImageDraw.Draw(img)
prev = None
for b in range(blocks):
    im = np.asarray(Image.open(f'{trace}/lens_s0_b{b:02d}.png').convert('RGB').resize((96, 96)), dtype=np.float32)
    img.paste(Image.fromarray(im.astype(np.uint8)), (b * 96, 30))
    if prev is not None:
        delta = np.abs(im - prev).mean(-1)
        delta = (delta / (delta.max() + 1e-6)) ** 0.7
        heat = (delta[..., None] * PLUS[None, None, :]).astype(np.uint8)
        img.paste(Image.fromarray(heat), (b * 96, 30 + 96 + 10))
    raw = Image.open(f'{trace}/rawlens_s0_b{b:02d}.png').convert('RGB').resize((96, 96))
    img.paste(raw, (b * 96, 30 + 192 + 20))
    prev = im
d.text((6, 6), 'step 1, blocks 1-25: what each block believes (tuned lens) / what that block changed / the old raw readout', fill=ASH, font=SMALL)
img.save(f'{OUT}/C_blocks_paint.png')
print('C written')
