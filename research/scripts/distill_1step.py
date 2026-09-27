"""Train the 1-step side branch on the ternary painter itself.

EPFL's RDM (epfl-vita/flux2-klein-1step-rdm) is a 1-step fine-tune of full-precision FLUX.2 klein 4B. Carried over to
the ternary painter as a low-rank side branch (rdm_delta.py, rdm_rank.py) it paints in 1 step, but softer than RDM
itself, with defects (a cat's face doubled). Here the branch is trained to close that gap by plain regression:

    teacher   RDM on klein, 1 step: x0 = z - v(z, t = 1)
    student   the ternary painter + a rank-R branch on its 100 big matrices (started from RDM's change, rdm_lora_r256)
              + the 1-step modulation (started from RDM's time and modulation weights, as schedule '1r'), trainable
    loss      LPIPS (VGG) between the two x0 decoded by TAEF2 (differentiable, as the browser decodes), plus a little
              mean squared error in the latent space to anchor colour; the same noise and prompt for both. (Mean
              squared error alone lowers the error by blurring: the student hedges where it cannot match.)

Both see the conditioning the browser gives: Ternary Bonsai 1.7B's layers 3, 6, 9 through the fitted map, 256 rows.

    python scripts/distill_1step.py targets N        # N prompts from adapter_train.txt: conditioning and teacher x0
    python scripts/distill_1step.py train STEPS [R]  # train (resumes from data/distill/student_rR.pt); the 12
                                                     # calibration prompts are held out and painted every 200 steps
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from PIL import Image
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
from rdm_delta_map import pieces  # noqa: E402

D = 'data/distill'
os.makedirs(D, exist_ok=True)
dev, dt = 'mps', torch.bfloat16
NT, TAPS = 256, (3, 6, 9)
HUB = '/Users/neo/.cache/huggingface/hub/'
TERN = glob.glob(HUB + 'models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
KLEIN = glob.glob(HUB + 'models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/')[0]
READER = glob.glob(HUB + 'models--prism-ml--Ternary-Bonsai-1.7B-unpacked/snapshots/*/')[0]
RDM = glob.glob(HUB + 'models--epfl-vita--flux2-klein-1step-rdm/snapshots/*/model.safetensors')[0]
LOWRANK = 'data/rdm/rdm_lora_r256.safetensors'
OTHER = 'data/rdm/rdm_other.safetensors'
MOD = ('time_guidance_embed.', 'double_stream_modulation_img.', 'double_stream_modulation_txt.',
       'single_stream_modulation.', 'norm_out.linear.')


# ---- conditioning, as the browser makes it
@torch.no_grad()
def conditioning(prompts, bs=8):
    import pack_model as pm
    tok = AutoTokenizer.from_pretrained(TERN + 'tokenizer')
    reader = AutoModelForCausalLM.from_pretrained(READER, dtype=torch.float32).to(dev).eval()
    reader.model.layers = reader.model.layers[:max(TAPS)]
    man = json.load(open(pm.PAINTER + '/manifest.json'))

    def dense(name):
        e = man['tensors'][name]
        dtp = np.float16 if e['dtype'] == 'f16' else np.float32
        return torch.from_numpy(np.fromfile(pm.PAINTER + '/' + e['file'], dtype=dtp, count=e['bytes'] // np.dtype(dtp).itemsize,
                                            offset=e['offset']).reshape(e['shape']).astype(np.float32))

    fused = torch.load('data/adapter/fused_final_layers_3_6_9.pt')
    W, b = fused['W'].to(dev), fused['b'].to(dev)
    prefix = (dense('adapter.prefix') @ dense('context_embedder.weight').T).to(dev)
    out = []
    for i in range(0, len(prompts), bs):
        text = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False) for p in prompts[i:i + bs]]
        inp = tok(text, return_tensors='pt', padding='max_length', max_length=NT, truncation=True).to(dev)
        hs = reader.model(**inp, output_hidden_states=True, use_cache=False).hidden_states
        ctx = torch.cat([hs[k].float() for k in TAPS], -1) @ W + b
        ctx[:, :3] = prefix
        out.append(ctx.half().cpu())
    del reader
    torch.mps.empty_cache()
    return torch.cat(out)


def painter(base):
    tr = Flux2Transformer2DModel.from_pretrained(base, subfolder='transformer', torch_dtype=dt)
    tr.context_embedder = nn.Identity()  # the conditioning is already in the painter's space
    return tr


def teacher():
    """RDM on klein: klein's transformer with every tensor RDM has."""
    tr = painter(KLEIN)
    params = dict(tr.named_parameters())
    with safe_open(RDM, 'pt') as f:
        for k in f.keys():
            full = f.get_tensor(k)
            for dk, sl in pieces(k):
                if dk == 'context_embedder.weight':
                    continue  # folded into the map
                if sl == 'swap':
                    h = full.shape[0] // 2
                    w1 = torch.cat([full[h:], full[:h]])
                elif sl is not None:
                    w1 = full[sl[0]:sl[1]]
                else:
                    w1 = full
                params[dk].data = w1.to(dt)
    return tr.to(dev).eval()


# ---- the student: side branches and the 1-step modulation, trainable
class Branch(nn.Module):
    def __init__(self, lin, A, B):
        super().__init__()
        self.lin, self.A, self.B = lin, nn.Parameter(A), nn.Parameter(B)

    def forward(self, x):
        return self.lin(x) + ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class Offset(nn.Module):
    """A module's output plus a learned constant (the modulation at t = 1 is one vector per module)."""

    def __init__(self, mod, n):
        super().__init__()
        self.mod, self.d = mod, nn.Parameter(torch.zeros(n))

    def forward(self, *a, **kw):
        return self.mod(*a, **kw) + self.d.to(dt)


def student(rank, state=None):
    tr = painter(TERN)
    params = dict(tr.named_parameters())
    with safe_open(OTHER, 'pt') as fo, safe_open(KLEIN + 'transformer/diffusion_pytorch_model.safetensors', 'pt') as fk:
        for k in fo.keys():  # RDM's change to the time and modulation weights, as schedule '1r' has it
            if k.startswith(MOD):
                params[k].data = (params[k].data.float() + fo.get_tensor(k).float() - fk.get_tensor(k).float()).to(dt)
    for p in tr.parameters():
        p.requires_grad_(False)
    branches = {}
    with safe_open(LOWRANK, 'pt') as s:
        for m in sorted({k.rsplit('.lora_', 1)[0] for k in s.keys()}):
            name = m.removeprefix('transformer.')
            parent, attr = name.rsplit('.', 1)
            lin = tr.get_submodule(name)
            br = Branch(lin, s.get_tensor(f'{m}.lora_A.weight').float()[:rank].clone(),
                        s.get_tensor(f'{m}.lora_B.weight').float()[:, :rank].clone())
            setattr(tr.get_submodule(parent), attr, br)
            branches[name] = br
    offsets = {}
    for name, n in (('double_stream_modulation_img', 6 * 3072), ('double_stream_modulation_txt', 6 * 3072),
                    ('single_stream_modulation', 3 * 3072)):
        off = Offset(getattr(tr, name), n)
        setattr(tr, name, off)
        offsets[name] = off
    off = Offset(tr.norm_out.linear, 2 * 3072)
    tr.norm_out.linear = off
    offsets['norm_out.linear'] = off
    if state:
        for name, br in branches.items():
            br.A.data, br.B.data = state['branches'][name]['A'], state['branches'][name]['B']
        for name, o in offsets.items():
            o.d.data = state['offsets'][name]
    tr.to(dev)
    return tr, branches, offsets


def x0(tr, z, ids, ctx, txt_ids):
    v = tr(hidden_states=z.to(dt), timestep=torch.ones(z.shape[0], device=dev, dtype=dt), guidance=None,
           encoder_hidden_states=ctx.to(dev, dt), txt_ids=txt_ids, img_ids=ids, return_dict=False)[0].float()
    return z - v


def noise(seed, pipe):
    z = torch.randn((1, 128, 32, 32), generator=torch.Generator('cpu').manual_seed(seed)).to(dev)
    return pipe._pack_latents(z), pipe._prepare_latent_ids(z).to(dev)


def decode(pipe, x):
    lat = x.permute(0, 2, 1).reshape(1, 128, 32, 32)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, torch.float32)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, torch.float32)
    with torch.no_grad():
        img = pipe.vae.decode(pipe._unpatchify_latents(lat * std + mean).to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type='pil')[0]


def taef2():
    """TAEF2's decoder in PyTorch (taesd.py's FLUX.2 variant): the unpatchified, normalised latent -> RGB 0..1."""
    import re
    from safetensors.torch import load_file
    from taesd import Decoder
    sd = load_file(glob.glob(HUB + 'models--madebyollin--taef2/snapshots/*/taef2.safetensors')[0])
    dec = Decoder(32, use_midblock_gn=True)
    dec.load_state_dict({re.sub(r'^decoder\.layers\.(\d+)', lambda m: str(int(m.group(1)) + 1), k): v
                         for k, v in sd.items() if k.startswith('decoder.')})
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec.to(dev).eval()


def base_pipe():
    pipe = Flux2KleinPipeline.from_pretrained(TERN, text_encoder=None, transformer=None, torch_dtype=dt)
    pipe.vae.to(dev)
    return pipe


# ---- the steps
@torch.no_grad()
def targets(n):
    prompts = [l.strip() for l in open('data/prompts/adapter_train.txt') if l.strip()][:n]
    held = [l.strip() for l in open('data/calibration_prompts.txt') if l.strip()]
    t0 = time.time()
    ctx = conditioning(prompts + held)
    print(f'conditioning: {time.time() - t0:.0f}s', flush=True)
    pipe = base_pipe()
    tr = teacher()
    txt_ids = pipe._prepare_text_ids(ctx[:1].to(dev)).to(dev)
    xs = []
    for i in range(len(ctx)):
        z, ids = noise(1000 + i, pipe)
        xs.append(x0(tr, z, ids, ctx[i:i + 1], txt_ids).half().cpu())
        if i % 50 == 0:
            print(f'teacher {i}/{len(ctx)} {time.time() - t0:.0f}s', flush=True)
    xs = torch.cat(xs)
    torch.save(dict(prompts=prompts, held=held, ctx=ctx[:n], x0=xs[:n]), f'{D}/train.pt')
    torch.save(dict(prompts=held, ctx=ctx[n:], x0=xs[n:]), f'{D}/held.pt')
    os.makedirs('renders/distill', exist_ok=True)
    for i in range(len(held)):
        decode(pipe, xs[n + i:n + i + 1].float().to(dev)).save(f'renders/distill/h{i}_teacher.png')
    print('saved', len(prompts), 'training prompts and', len(held), 'held out', f'{time.time() - t0:.0f}s')


def train(steps, rank, mse_weight=0.05):
    tr_data = torch.load(f'{D}/train.pt')
    held = torch.load(f'{D}/held.pt')
    n = len(tr_data['prompts'])
    pipe = base_pipe()
    state = torch.load(f'{D}/student_r{rank}.pt') if os.path.exists(f'{D}/student_r{rank}.pt') else None
    tr, branches, offsets = student(rank, state)
    tr.enable_gradient_checkpointing()
    txt_ids = pipe._prepare_text_ids(tr_data['ctx'][:1].to(dev)).to(dev)
    import lpips
    dec = taef2()
    percept = lpips.LPIPS(net='vgg', verbose=False).to(dev).eval()
    for p in percept.parameters():
        p.requires_grad_(False)

    def pixels(x):  # packed latent -> image in [-1, 1]
        return dec(pipe._unpatchify_latents(x.permute(0, 2, 1).reshape(-1, 128, 32, 32))).clamp(0, 1) * 2 - 1

    opt = torch.optim.AdamW([{'params': [p for b in branches.values() for p in (b.A, b.B)], 'lr': 5e-5},
                             {'params': [o.d for o in offsets.values()], 'lr': 1e-4}], weight_decay=0.0)
    start = state['step'] if state else 0
    log = []

    def held_loss(save=None):
        tr.eval()
        err = []
        with torch.no_grad():
            for i in range(len(held['prompts'])):
                z, ids = noise(1000 + n + i, pipe)
                xs = x0(tr, z, ids, held['ctx'][i:i + 1], txt_ids)
                err.append(((xs - held['x0'][i:i + 1].float().to(dev)) ** 2).mean().item())
                if save:
                    decode(pipe, xs).save(f'renders/distill/h{i}_{save}.png')
        tr.train()
        return sum(err) / len(err)

    if not state:
        print(f'held-out error before training: {held_loss(save=f"r{rank}_start"):.5f}', flush=True)
    tr.train()
    t0 = time.time()
    order = torch.randperm(n, generator=torch.Generator().manual_seed(start))
    for s in range(start, start + steps):
        i = int(order[s % n])
        z, ids = noise(1000 + i, pipe)
        xs = x0(tr, z, ids, tr_data['ctx'][i:i + 1], txt_ids)
        xt = tr_data['x0'][i:i + 1].float().to(dev)
        with torch.no_grad():
            target = pixels(xt)
        loss = percept(pixels(xs), target).mean() + mse_weight * ((xs - xt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log.append(loss.item())
        if (s + 1) % 10 == 0:
            print(f'step {s + 1}: loss {sum(log[-10:]) / 10:.5f}, {(time.time() - t0) / (s + 1 - start):.1f}s/step',
                  flush=True)
        if (s + 1) % 100 == 0 or s + 1 == start + steps:
            h = held_loss(save=f'r{rank}_s{s + 1}')
            print(f'step {s + 1}: held-out error {h:.5f}', flush=True)
            torch.save(dict(step=s + 1, rank=rank,
                            branches={k: dict(A=b.A.data.cpu(), B=b.B.data.cpu()) for k, b in branches.items()},
                            offsets={k: o.d.data.cpu() for k, o in offsets.items()}), f'{D}/student_r{rank}.pt')


def export(rank):
    """The trained branch as a LoRA file for pack_model.py --rdm (and the offsets for export_painter_schedules.py)."""
    from safetensors.torch import save_file
    st = torch.load(f'{D}/student_r{rank}.pt')
    out = {}
    for name, ab in st['branches'].items():
        out[f'transformer.{name}.lora_A.weight'] = ab['A'].to(torch.bfloat16).contiguous()
        out[f'transformer.{name}.lora_B.weight'] = ab['B'].to(torch.bfloat16).contiguous()
    save_file(out, f'{D}/rdm_trained_r{rank}.safetensors', metadata={'rank': str(rank), 'alpha': str(rank), 'step': str(st['step'])})
    print(f'wrote {D}/rdm_trained_r{rank}.safetensors (step {st["step"]}); then:')
    print(f'  RDM_STUDENT={D}/student_r{rank}.pt python scripts/export_painter_schedules.py')
    print(f'  python scripts/pack_model.py ... --rdm {D}/rdm_trained_r{rank}.safetensors --rdm-rank {rank}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'targets':
        targets(int(sys.argv[2]))
    elif cmd == 'train':
        train(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 32)
    elif cmd == 'export':
        export(int(sys.argv[2]) if len(sys.argv) > 2 else 32)
