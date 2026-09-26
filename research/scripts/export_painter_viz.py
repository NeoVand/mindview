"""Readouts the live piece uses to show the painter's current guess at every block (not needed to paint).

1. The tuned lens (data/tuned_lens/lens.pt): per block, an affine map from an image token's 3072-d state to that step's
   velocity (128). With it, x0 ~ x_t - sigma * v_lens is the painting the block 'has in mind'.
2. A latent-pixel colour probe: a linear map from one latent pixel's 32 channels (bn-normalised space) to the mean RGB
   of the 8x8 image pixels it decodes to. Fitted on images encoded by TAEF2's own encoder.

Writes static/models/bonsai-image-4b/viz.bin (+ viz.json): lens f16 [25, 3073, 128] (last row = bias), probe f32 [33, 3].
"""
import glob, json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
sys.path.insert(0, os.path.dirname(__file__))
from taesd import Encoder

OUT = '../static/models/bonsai-image-4b'
torch.set_grad_enabled(False)

# ---- TAEF2 encoder (its latents live in the same bn-normalised space the decoder reads)
sd = load_file(hf_hub_download('madebyollin/taef2', 'taef2.safetensors'))
enc = Encoder(32, use_midblock_gn=True)
enc.load_state_dict({k[len('encoder.layers.'):]: v for k, v in sd.items() if k.startswith('encoder.layers.')})
enc.eval()


def encode(img):  # PIL 512x512 -> latent [32, 64, 64]
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)[None]
    return enc(x)[0]


# sanity: the encoder should land near the painter's own final latent for its own picture
ref = json.load(open('../static/lab/painter_ref/ref.json'))
lat = np.fromfile('../static/lab/painter_ref/latents.bin', dtype='<f4').reshape(4, 1024, 128)[3]
z = torch.from_numpy(lat).view(32, 32, 32, 2, 2).permute(2, 0, 3, 1, 4).reshape(32, 64, 64)
e = encode(Image.open('../static/lab/painter_ref/final.png').convert('RGB'))
print('TAEF2 encoder vs the painter latent: cosine', F.cosine_similarity(e.flatten(), z.flatten(), 0).item())

# ---- fit: latent pixel (32) -> mean RGB of its 8x8 block
paths = [p for p in glob.glob('renders/**/*.png', recursive=True) + glob.glob('../static/**/*.png', recursive=True)]
random.seed(0)
X, Y = [], []
for p in paths:
    try:
        im = Image.open(p).convert('RGB')
    except Exception:
        continue
    if min(im.size) < 256:
        continue
    for _ in range(3 if min(im.size) >= 768 else 1):
        s = min(im.size)
        c = random.randint(512, s) if s > 512 else s
        x0, y0 = random.randint(0, im.size[0] - c), random.randint(0, im.size[1] - c)
        crop = im.crop((x0, y0, x0 + c, y0 + c)).resize((512, 512), Image.BICUBIC)
        l = encode(crop)
        t = F.avg_pool2d(torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255).permute(2, 0, 1)[None], 8)[0]
        X.append(l.reshape(32, -1).T)
        Y.append(t.reshape(3, -1).T)
X, Y = torch.cat(X).double(), torch.cat(Y).double()
Xb = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], 1)
W = torch.linalg.solve(Xb.T @ Xb + 1e-2 * torch.eye(33, dtype=torch.float64), Xb.T @ Y)
pred = Xb @ W
r2 = 1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
print(f'probe fitted on {len(X)} latent pixels from {len(paths)} files; R^2 per channel', r2.tolist())
# check on the painter's own picture
img = torch.from_numpy(np.asarray(Image.open('../static/lab/painter_ref/final.png').convert('RGB'), dtype=np.float32) / 255)
t = F.avg_pool2d(img.permute(2, 0, 1)[None], 8)[0].reshape(3, -1).T.double()
pz = torch.cat([z.reshape(32, -1).T.double(), torch.ones(4096, 1, dtype=torch.float64)], 1) @ W
print('on the painter latent: mean abs error', (pz - t).abs().mean().item())

# ---- write
L = torch.load('data/tuned_lens/lens.pt', map_location='cpu')['W']  # [25, 3073, 128]
lens = L.to(torch.float16).numpy()
probe = W.float().numpy()
with open(f'{OUT}/viz.bin', 'wb') as f:
    f.write(lens.tobytes())
    f.write(probe.astype('<f4').tobytes())
json.dump(dict(lens=dict(dtype='f16', shape=list(lens.shape), offset=0, bytes=lens.nbytes,
                         note='v = [h, 1] @ W[block]; x0 = x_t - sigma * v'),
               probe=dict(dtype='f32', shape=[33, 3], offset=lens.nbytes, bytes=probe.nbytes,
                          note='rgb = [latent pixel (32 channels), 1] @ W, bn-normalised latent')),
          open(f'{OUT}/viz.json', 'w'), indent=1)
print('wrote', f'{OUT}/viz.bin', os.path.getsize(f'{OUT}/viz.bin') / 1e6, 'MB')
