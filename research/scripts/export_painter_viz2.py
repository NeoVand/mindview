"""A better per-block picture for the live piece: TAEF2's first stage (64 x 64, three residual blocks) already turns a
noisy latent into clean image features; a linear probe then reads colour off those 64 channels.

Fits probe64 [65, 3]: rgb (display space) = [features after taef2 layer 4 (64 ch), 1] @ W, against TAEF2's own full
decode averaged over each 8 x 8 pixel block. Latents come from TAEF2's encoder on images in the repo.
Rewrites static/models/bonsai-image-4b/viz.bin/json with lens + probe (32 -> rgb) + probe64.
"""
import glob, json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
sys.path.insert(0, os.path.dirname(__file__))
from taesd import Encoder, Decoder

OUT = '../static/models/bonsai-image-4b'
torch.set_grad_enabled(False)
dev = 'mps'
sd = load_file(hf_hub_download('madebyollin/taef2', 'taef2.safetensors'))
enc = Encoder(32, use_midblock_gn=True)
enc.load_state_dict({k[len('encoder.layers.'):]: v for k, v in sd.items() if k.startswith('encoder.layers.')})
dec = Decoder(32, use_midblock_gn=True)
dec.load_state_dict({('%d.' % (int(k.split('.')[2]) + 1)) + '.'.join(k.split('.')[3:]): v for k, v in sd.items() if k.startswith('decoder.layers.')})
enc, dec = enc.to(dev).eval(), dec.to(dev).eval()
stage = dec[:6]  # Clamp, conv, ReLU, 3 gn blocks (taef2 layers 0..4) -> [64, 64, 64]

paths = sorted(glob.glob('renders/**/*.png', recursive=True))
random.seed(1)
random.shuffle(paths)
X, Y = [], []
for p in paths[:260]:
    try:
        im = Image.open(p).convert('RGB')
    except Exception:
        continue
    if min(im.size) < 256:
        continue
    s = min(im.size)
    c = random.randint(min(512, s), s)
    x0, y0 = random.randint(0, im.size[0] - c), random.randint(0, im.size[1] - c)
    crop = im.crop((x0, y0, x0 + c, y0 + c)).resize((512, 512), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(dev)
    z = enc(x)
    # the live piece shows noisy guesses too: train on clean latents and on latents with some noise left in
    for noise in (0.0, 0.3, 0.6):
        zz = z + noise * torch.randn_like(z)
        f = stage(zz)[0]                                          # [64, 64, 64]
        img = dec(zz).clamp(0, 1)                                 # TAEF2's own picture of that latent
        t = F.avg_pool2d(img, 8)[0]                               # [3, 64, 64]
        X.append(f.reshape(64, -1).T.cpu().double())
        Y.append(t.reshape(3, -1).T.cpu().double())
X, Y = torch.cat(X), torch.cat(Y)
Xb = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], 1)
W = torch.linalg.solve(Xb.T @ Xb + 1e-2 * torch.eye(65, dtype=torch.float64), Xb.T @ Y)
r2 = 1 - ((Xb @ W - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
print(f'probe64 on {len(X)} pixels: R^2', [round(v, 4) for v in r2.tolist()])

# check on the painter's own final latent
lat = np.fromfile('../static/lab/painter_ref/latents.bin', dtype='<f4').reshape(4, 1024, 128)[3]
z = torch.from_numpy(lat).view(32, 32, 32, 2, 2).permute(2, 0, 3, 1, 4).reshape(1, 32, 64, 64).to(dev)
f = stage(z)[0].reshape(64, -1).T.cpu().double()
t = F.avg_pool2d(dec(z).clamp(0, 1), 8)[0].reshape(3, -1).T.cpu().double()
print('on the painter latent: mean abs error', (torch.cat([f, torch.ones(4096, 1, dtype=torch.float64)], 1) @ W - t).abs().mean().item())

# rewrite viz.bin: lens (f16), probe (f32 [33,3]) as before, then probe64 (f32 [65,3])
meta = json.load(open(f'{OUT}/viz.json'))
raw = open(f'{OUT}/viz.bin', 'rb').read()
keep = raw[: meta['probe']['offset'] + meta['probe']['bytes']]
p64 = W.float().numpy().astype('<f4')
with open(f'{OUT}/viz.bin', 'wb') as fh:
    fh.write(keep)
    fh.write(p64.tobytes())
meta['probe64'] = dict(dtype='f32', shape=[65, 3], offset=len(keep), bytes=p64.nbytes,
                       note='rgb = [TAEF2 features after layer 4 (64 ch at 64x64), 1] @ W')
json.dump(meta, open(f'{OUT}/viz.json', 'w'), indent=1)
print('viz.bin', os.path.getsize(f'{OUT}/viz.bin') / 1e6, 'MB')
