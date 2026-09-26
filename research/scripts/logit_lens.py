"""Logit lens on captured residual streams: what would the model say if it stopped at layer k?"""
import sys, glob, json
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

repo = sys.argv[1] if len(sys.argv) > 1 else 'Ternary-Bonsai-1.7B-unpacked'
path = glob.glob(f'/Users/neo/.cache/huggingface/hub/models--prism-ml--{repo}/snapshots/*/')[0]
tok = AutoTokenizer.from_pretrained(path)
d = np.load(f'data/capture_{repo}.npz'); meta = json.load(open(f'data/capture_{repo}.json'))
with safe_open(path + 'model.safetensors', 'np') as f:
    E = f.get_tensor('model.embed_tokens.weight').astype(np.float32)
    nw = f.get_tensor('model.norm.weight').astype(np.float32)
H = d['hidden']  # (T, L+1, d); index L is already final-normed in HF hidden_states
T, L1, _ = H.shape
for t in [2, 5, 13, 20]:
    print(f"\n== step {t}: model emitted {meta['text'][t]!r}")
    for layer in list(range(0, L1 - 1, 3)) + [L1 - 2]:
        h = H[t, layer]
        hn = h / np.sqrt((h ** 2).mean() + 1e-6) * nw
        logits = E @ hn
        top = np.argsort(-logits)[:4]
        p = np.exp(logits - logits.max()); p /= p.sum()
        print(f"  L{layer:2d}: " + ' | '.join(f'{tok.decode([int(i)])!r} {p[i]:.2f}' for i in top))
