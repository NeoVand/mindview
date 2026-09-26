"""Show how the VAE decoder actually forms pixels: latent (64x64) -> 128 -> 256 -> 512.

For each decoder stage we fit a linear readout (channels -> RGB, ridge, across ~30 images) so the stage's feature
map can be shown as the picture it holds at that resolution. The last stage is the real output.

  decoder_stages.py fit
  decoder_stages.py render <trace_dir>      # writes <trace_dir>/dec_stage{k}.png from the trace's final latent
"""
import sys, glob, os
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from diffusers import AutoencoderKLFlux2

dev = 'mps'
ROOT = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
vae = AutoencoderKLFlux2.from_pretrained(ROOT, subfolder='vae', torch_dtype=torch.float32).to(dev).eval()
dec = vae.decoder
STAGES = [('mid', dec.mid_block), ('up0', dec.up_blocks[0]), ('up1', dec.up_blocks[1]), ('up2', dec.up_blocks[2]), ('up3', dec.up_blocks[3])]
feats = {}
for name, mod in STAGES:
    mod.register_forward_hook(lambda m, a, o, name=name: feats.__setitem__(name, o[0] if isinstance(o, tuple) else o))
OUT = 'data/decoder_probe.pt'


def load_img(p):
    x = torch.from_numpy(np.asarray(Image.open(p).convert('RGB').resize((512, 512)), dtype=np.float32).copy()).permute(2, 0, 1)[None] / 127.5 - 1
    return x.to(dev)


mode = sys.argv[1]
if mode == 'fit':
    paths = sorted(glob.glob('renders/image/compare*/*.png'))
    paths = [p for p in paths if 'grid' not in p][:40]
    stats = {}
    with torch.no_grad():
        for p in paths:
            img = load_img(p)
            lat = vae.encode(img).latent_dist.mean
            vae.decode(lat)
            for name, _ in STAGES:
                f = feats[name][0]                                   # [C, H, W]
                C, H, W = f.shape
                y = Fn.interpolate(img, size=(H, W), mode='area')[0]  # [3, H, W] target at this resolution
                X = torch.cat([f.reshape(C, -1).T, torch.ones(H * W, 1, device=dev)], 1)
                Y = y.reshape(3, -1).T
                s = stats.setdefault(name, [torch.zeros(C + 1, C + 1, device=dev), torch.zeros(C + 1, 3, device=dev), 0])
                s[0] += X.T @ X; s[1] += X.T @ Y; s[2] += X.shape[0]
    probes = {}
    for name, (xtx, xty, n) in stats.items():
        A = xtx.cpu().double() / n
        B = xty.cpu().double() / n
        reg = 1e-3 * torch.diag(torch.diag(A)).clamp_min(1e-8)
        probes[name] = torch.linalg.solve(A + reg, B).float()
        print(name, 'channels', A.shape[0] - 1)
    torch.save(probes, OUT)
    print('fitted on', len(paths), 'images')

elif mode == 'render':
    d = sys.argv[2]
    probes = torch.load(OUT)
    with torch.no_grad():
        # the trace stores the final image; re-encode it to get its latent (a close stand-in for the true latent)
        lat_path = os.path.join(d, 'final_latent.npy')
        if os.path.exists(lat_path):
            lat = torch.from_numpy(np.load(lat_path)).to(dev)
        else:
            lat = vae.encode(load_img(os.path.join(d, 'final.png'))).latent_dist.mean
        out = vae.decode(lat).sample
        for k, (name, _) in enumerate(STAGES):
            f = feats[name][0]
            C, H, W = f.shape
            X = torch.cat([f.reshape(C, -1).T, torch.ones(H * W, 1, device=dev)], 1)
            rgb = (X @ probes[name].to(dev)).T.reshape(3, H, W)
            im = ((rgb.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(im).save(os.path.join(d, f'dec_stage{k}.png'))
            print(name, (H, W))
        # the latent itself, as the decoder receives it (first 3 principal directions of its 32 channels)
        z = lat[0]
        zc = z.reshape(32, -1).T
        zc = zc - zc.mean(0)
        V = torch.linalg.eigh((zc.T @ zc).cpu().double())[1][:, -3:].flip(-1).float().to(dev)
        p = zc @ V
        lo, hi = torch.quantile(p, 0.01, 0), torch.quantile(p, 0.99, 0)
        p = ((p - lo) / (hi - lo)).clamp(0, 1).T.reshape(3, z.shape[1], z.shape[2])
        Image.fromarray((p * 255).byte().permute(1, 2, 0).cpu().numpy()).save(os.path.join(d, 'dec_latent.png'))
        final = ((out[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(final).save(os.path.join(d, f'dec_stage{len(STAGES)}.png'))
    # tell the player which frames to show during decoding, in order (up3 is skipped: same size as up2)
    import json
    mpath = os.path.join(d, 'manifest.json')
    m = json.load(open(mpath))
    m['decoder'] = [
        dict(file='dec_latent.png', res=64, label='The painter hands over a 64 × 64 grid of 32 numbers each'),
        dict(file='dec_stage0.png', res=64, label='Decoding into pixels: 64 × 64'),
        dict(file='dec_stage1.png', res=128, label='Decoding into pixels: 128 × 128'),
        dict(file='dec_stage2.png', res=256, label='Decoding into pixels: 256 × 256'),
        dict(file='dec_stage3.png', res=512, label='Decoding into pixels: 512 × 512'),
        dict(file=f'dec_stage{len(STAGES)}.png', res=512, label='The image'),
    ]
    json.dump(m, open(mpath, 'w'))
    print('wrote decoder stages to', d)
