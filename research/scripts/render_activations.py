"""Render captured activations: MLP neuron field per token, residual stream, and ternary 'contribution' maps."""
import sys, glob, json
import numpy as np
from PIL import Image
from safetensors import safe_open

repo = sys.argv[1] if len(sys.argv) > 1 else 'Ternary-Bonsai-1.7B-unpacked'
d = np.load(f'data/capture_{repo}.npz')
meta = json.load(open(f'data/capture_{repo}.json'))
neurons = d['neurons']  # (T, L, 6144)
hidden = d['hidden']    # (T, L+1, 2048)
T, L, F = neurons.shape
print('tokens', T, 'layers', L, 'ffn', F)

def diverge(a, gamma=0.5):
    """signed values -> cyan/black/amber"""
    m = np.quantile(np.abs(a), 0.995) + 1e-9
    v = np.clip(a / m, -1, 1)
    mag = np.abs(v) ** gamma
    out = np.zeros(a.shape + (3,), np.float32)
    pos = v > 0
    out[pos] = mag[pos, None] * np.array([255, 170, 40])
    out[~pos] = mag[~pos, None] * np.array([40, 200, 255])
    return out.astype(np.uint8)

# 1) neuron field for one token: 28 rows x 6144 cols, reshaped to 28 x (48x128) tiles -> a "brain slice"
t = 5
fld = neurons[t]  # (L, F)
img = diverge(fld)  # (L, F, 3)
img = np.repeat(img, 24, axis=0)
Image.fromarray(img).save(f'renders/neurons_token{t}_{meta["text"][t].strip()}.png')

# 2) neuron sparsity/intensity over the whole generation: rows = token, cols = layer*F (downsampled)
energy = np.abs(neurons).mean(2)  # (T, L)
print('mean |neuron| per layer (avg over tokens):', np.round(energy.mean(0), 3).tolist())
act_frac = (np.abs(neurons) > 0.1 * np.abs(neurons).max(2, keepdims=True)).mean(2)
print('fraction of neurons > 10% of layer max:', np.round(act_frac.mean(0), 3).tolist())

# 3) residual stream per layer per token, normalized per layer (RMS) to see directions not magnitudes
h = hidden / (np.sqrt((hidden ** 2).mean(-1, keepdims=True)) + 1e-6)
strip = diverge(h.transpose(1, 0, 2).reshape((L + 1), T * hidden.shape[2])[:, : T * 2048])
top_dims = np.argsort(-np.abs(hidden[:, 1:-1]).mean((0, 1)))[:8]
print('massive residual dims (largest mean |h|):', top_dims.tolist(),
      np.round(np.abs(hidden[:, 1:-1]).mean((0, 1))[top_dims], 1).tolist(), 'vs median', np.round(np.median(np.abs(hidden[:, 1:-1])), 2))
res_img = diverge(h[:, :, :].transpose(1, 0, 2).reshape(L + 1, -1))
# per token: a (L+1) x 2048 image
for tt in [0, 5, 20]:
    Image.fromarray(np.repeat(diverge(h[tt]), 16, 0)).save(f'renders/residual_token{tt}.png')

# 4) contribution map of a ternary matrix for one token: C_ij = t_ij * x_j (which synapses carried signal)
path = glob.glob(f'/Users/neo/.cache/huggingface/hub/models--prism-ml--{repo}/snapshots/*/model.safetensors')[0]
with safe_open(path, 'np') as f:
    w = f.get_tensor('model.layers.14.mlp.down_proj.weight').astype(np.float32)  # (2048, 6144)
x = neurons[t, 14]  # input to down_proj
C = w * x[None, :]
crop = C[:512, np.argsort(-np.abs(x))[:1024]]  # the 1024 most active neurons
Image.fromarray(diverge(crop, 0.4)).save('renders/contrib_L14_down_top1024.png')
Image.fromarray(diverge(C[:512, :1024], 0.4)).save('renders/contrib_L14_down_natural.png')
print('down_proj L14: |x| top-1% neurons carry', round(float(np.sort(np.abs(x))[::-1][: F // 100].sum() / np.abs(x).sum()), 3), 'of total |x| mass')
