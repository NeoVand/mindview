"""Render ternary weight matrices as images: -1 -> cyan, 0 -> black, +1 -> amber."""
import sys, glob
import numpy as np
from PIL import Image
from safetensors import safe_open

repo = sys.argv[1]
path = glob.glob(f'/Users/neo/.cache/huggingface/hub/models--prism-ml--{repo}/snapshots/*/model.safetensors')[0]
G = 128
PAL = np.array([[40, 200, 255], [0, 0, 0], [255, 170, 40]], np.uint8)  # -1, 0, +1

def trits(w):
    w = w.astype(np.float32)
    r, c = w.shape
    g = w.reshape(r, c // G, G)
    s = np.abs(g).max(2, keepdims=True); s[s == 0] = 1
    return np.rint(g / s).astype(np.int8).reshape(r, c), s.squeeze(-1)

with safe_open(path, 'np') as f:
    for name in ['model.layers.0.self_attn.q_proj.weight', 'model.layers.14.mlp.gate_proj.weight',
                 'model.layers.14.self_attn.o_proj.weight', 'model.layers.27.mlp.down_proj.weight', 'model.embed_tokens.weight']:
        t, s = trits(f.get_tensor(name))
        crop = t[:1024, :1024]
        Image.fromarray(PAL[crop + 1]).save(f'renders/{repo}_{name.replace("model.", "").replace(".weight", "")}_crop1024.png')
        # full matrix, downsampled: mean trit (net polarity) and density of non-zero per 8x8 tile
        r, c = t.shape
        k = max(1, max(r, c) // 2048)
        tt = t[: r // k * k, : c // k * k].reshape(r // k, k, c // k, k).astype(np.float32)
        pol = tt.mean((1, 3)); dens = (tt != 0).mean((1, 3))
        img = np.zeros(pol.shape + (3,), np.float32)
        img += np.clip(pol, 0, None)[..., None] * PAL[2] * 4 + np.clip(-pol, 0, None)[..., None] * PAL[0] * 4
        img += ((dens - dens.mean()) / (dens.std() + 1e-6) * 40 + 60)[..., None]
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(f'renders/{repo}_{name.replace("model.", "").replace(".weight", "")}_full_k{k}.png')
        # group-scale map
        sm = (s - s.min()) / (s.max() - s.min() + 1e-9)
        Image.fromarray((sm * 255).astype(np.uint8)).resize((min(2048, s.shape[1] * 8), min(2048, s.shape[0])), Image.NEAREST).save(
            f'renders/{repo}_{name.replace("model.", "").replace(".weight", "")}_scales.png')
        print(name, t.shape, 'zero frac', (t == 0).mean().round(3))
