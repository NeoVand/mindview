"""Colour probes for the decoder's later stages, so the live piece can show the picture growing through TAEF2:
64 x 64 (probe64, from export_painter_viz2.py) -> 128 x 128 -> 256 x 256 -> 512 x 512 (the decoder's own output).

Fits probe128 and probe256 [65, 3]: rgb (display space) = [features after the stage's residual blocks (64 ch), 1] @ W,
against TAEF2's own full decode averaged over each 4 x 4 (resp. 2 x 2) pixel block. Latents come from TAEF2's encoder
on images in the repo, clean and with some noise left in (the live piece decodes the painter's guesses). The normal
equations are accumulated image by image (the 256 x 256 features are large).
Appends both to static/models/bonsai-image-4b/viz.bin / viz.json (keeps lens, probe, probe64).
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
# decoder: 0 Clamp, 1 conv, 2 ReLU, 3-5 blocks (64), 6 Up, 7 conv, 8-10 blocks (128), 11 Up, 12 conv, 13-15 blocks (256),
# 16 Up, 17 conv, 18 block (512), 19 conv -> rgb
STAGES = {128: 11, 256: 16}
print([type(m).__name__ for m in dec])

paths = sorted(glob.glob('renders/**/*.png', recursive=True))
random.seed(1)
random.shuffle(paths)
acc = {r: [torch.zeros(65, 65, dtype=torch.float64), torch.zeros(65, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 0] for r in STAGES}
used = 0
for p in paths:
    if used >= 160:
        break
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
    used += 1
    for noise in (0.0, 0.3, 0.6):
        zz = z + noise * torch.randn_like(z)
        img = dec(zz).clamp(0, 1)
        for res, upto in STAGES.items():
            f = dec[:upto](zz)[0]                                   # [64, res, res]
            t = F.avg_pool2d(img, 512 // res)[0]                    # [3, res, res]
            X = f.reshape(64, -1).T.cpu().double()
            Y = t.reshape(3, -1).T.cpu().double()
            Xb = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], 1)
            a = acc[res]
            a[0] += Xb.T @ Xb
            a[1] += Xb.T @ Y
            a[2] += Y.sum(0)
            a[3] += (Y ** 2).sum(0)
            a[4] += len(Y)
print('images used', used)

probes = {}
for res, (XtX, XtY, ys, yss, n) in acc.items():
    W = torch.linalg.solve(XtX + 1e-2 * torch.eye(65, dtype=torch.float64), XtY)
    # R^2 from the accumulated sums: residual = Y'Y - 2 W'X'Y + W'X'XW
    yy = yss
    res_ss = yy - 2 * (W * XtY).sum(0) + (W * (XtX @ W)).sum(0)
    tot = yss - ys ** 2 / n
    print(f'probe{res}: R^2', [round(v, 4) for v in (1 - res_ss / tot).tolist()])
    probes[res] = W.float().numpy().astype('<f4')

# check on the painter's own final latent
lat = np.fromfile('../static/lab/painter_ref/latents.bin', dtype='<f4').reshape(4, 1024, 128)[3]
z = torch.from_numpy(lat).view(32, 32, 32, 2, 2).permute(2, 0, 3, 1, 4).reshape(1, 32, 64, 64).to(dev)
img = dec(z).clamp(0, 1)
for res, upto in STAGES.items():
    f = dec[:upto](z)[0].reshape(64, -1).T.cpu().double()
    t = F.avg_pool2d(img, 512 // res)[0].reshape(3, -1).T.cpu().double()
    W = torch.from_numpy(probes[res]).double()
    print(f'probe{res} on the painter latent: mean abs error', (torch.cat([f, torch.ones(len(f), 1, dtype=torch.float64)], 1) @ W - t).abs().mean().item())

meta = json.load(open(f'{OUT}/viz.json'))
raw = open(f'{OUT}/viz.bin', 'rb').read()
keep = raw[: meta['probe64']['offset'] + meta['probe64']['bytes']]
blob = keep
for res in STAGES:
    meta[f'probe{res}'] = dict(dtype='f32', shape=[65, 3], offset=len(blob), bytes=probes[res].nbytes,
                                note=f'rgb = [TAEF2 features after the {res} x {res} stage (64 ch), 1] @ W')
    blob += probes[res].tobytes()
open(f'{OUT}/viz.bin', 'wb').write(blob)
json.dump(meta, open(f'{OUT}/viz.json', 'w'), indent=1)
print('viz.bin', os.path.getsize(f'{OUT}/viz.bin') / 1e6, 'MB')
