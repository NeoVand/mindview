"""Train the 1-step side branch on the ternary painter itself.

EPFL's RDM (epfl-vita/flux2-klein-1step-rdm) is a 1-step fine-tune of full-precision FLUX.2 klein 4B. Carried over to
the ternary painter as a low-rank side branch (rdm_delta.py, rdm_rank.py) it paints in 1 step, but softer than RDM
itself, with defects (a cat's face doubled). Here the branch is trained to close that gap:

    teacher   RDM on klein, 1 step: x0 = z - v(z, t = 1)
    student   the ternary painter + a rank-R branch on its 100 big matrices (started from RDM's change) + offsets on
              the 1-step modulation (started from RDM's time and modulation weights, as schedule '1r'), trainable
    loss      LPIPS (VGG) between the two x0 decoded by TAEF2 (differentiable, as the browser decodes), plus a little
              mean squared error in the latent space to anchor colour; the same noise and prompt for both. (Mean
              squared error alone lowers the error by blurring: the student hedges where it cannot match.)

Both see the conditioning the browser gives: Ternary Bonsai 1.7B's layers 3, 6, 9 through the fitted map, 256 rows.
The 12 calibration prompts are held out and painted as training goes (renders/distill/h<i>_<label>.png).

    python scripts/distill_1step.py prepare            # locally: the map (cond.pt) and the branch's start (init.pt)
    python scripts/distill_1step.py targets N          # N prompts from adapter_train.txt: conditioning, teacher x0
    python scripts/distill_1step.py train STEPS [R] [B] [LR]  # train (resumes from student_rR.pt), batch B
    python scripts/distill_1step.py export [R]         # the branch for pack_model.py --rdm

WORK (default: research/) holds data/ and renders/; HUB is the Hugging Face cache (default: ~/.cache/huggingface/hub).
Runs on CUDA or MPS (see research/cloud/distill_modal.py for the cloud run).
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from rdm_delta_map import pieces  # noqa: E402

os.chdir(os.environ.get('WORK', os.path.join(SCRIPTS, '..')))
D = 'data/distill'
os.makedirs(D, exist_ok=True)
os.makedirs('renders/distill', exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'mps'
dt = torch.bfloat16
NT, TAPS = 256, (3, 6, 9)
HUB = os.environ.get('HUB', os.path.expanduser('~/.cache/huggingface/hub')) + '/'


def snap(repo, f=''):
    return glob.glob(HUB + f'models--{repo.replace("/", "--")}/snapshots/*/' + f)[0]


TERN = lambda: snap('prism-ml/bonsai-image-ternary-4B-unpacked')  # noqa: E731
KLEIN = lambda: snap('black-forest-labs/FLUX.2-klein-4B')  # noqa: E731
READER = lambda: snap('prism-ml/Ternary-Bonsai-1.7B-unpacked')  # noqa: E731
RDM = lambda: snap('epfl-vita/flux2-klein-1step-rdm', 'model.safetensors')  # noqa: E731
MOD = ('time_guidance_embed.', 'double_stream_modulation_img.', 'double_stream_modulation_txt.',
       'single_stream_modulation.', 'norm_out.linear.')


# ---- inputs made locally (small): the map to the painter's text rows, and where the branch starts
def prepare(max_rank=64):
    sys.path.insert(0, SCRIPTS)
    import pack_model as pm
    man = json.load(open(pm.PAINTER + '/manifest.json'))

    def dense(name):
        e = man['tensors'][name]
        dtp = np.float16 if e['dtype'] == 'f16' else np.float32
        return torch.from_numpy(np.fromfile(pm.PAINTER + '/' + e['file'], dtype=dtp, count=e['bytes'] // np.dtype(dtp).itemsize,
                                            offset=e['offset']).reshape(e['shape']).astype(np.float32))

    fused = torch.load('data/adapter/fused_final_layers_3_6_9.pt')
    torch.save(dict(W=fused['W'].float(), b=fused['b'].float(),
                    prefix=dense('adapter.prefix') @ dense('context_embedder.weight').T), f'{D}/cond.pt')
    init = {}
    with safe_open('data/rdm/rdm_lora_r256.safetensors', 'pt') as s:  # rdm_delta.py: RDM - klein, SVD per matrix
        for m in sorted({k.rsplit('.lora_', 1)[0] for k in s.keys()}):
            init[m.removeprefix('transformer.')] = dict(A=s.get_tensor(f'{m}.lora_A.weight')[:max_rank].clone(),
                                                        B=s.get_tensor(f'{m}.lora_B.weight')[:, :max_rank].clone())
    torch.save(init, f'{D}/init.pt')
    print(f'wrote {D}/cond.pt and {D}/init.pt (rank up to {max_rank}, {len(init)} matrices)')


# ---- conditioning, as the browser makes it
@torch.no_grad()
def conditioning(prompts, bs=16):
    tok = AutoTokenizer.from_pretrained(TERN() + 'tokenizer')
    reader = AutoModelForCausalLM.from_pretrained(READER(), dtype=torch.float32).to(dev).eval()
    reader.model.layers = reader.model.layers[:max(TAPS)]
    c = torch.load(f'{D}/cond.pt')
    W, b, prefix = c['W'].to(dev), c['b'].to(dev), c['prefix'].to(dev)
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
    return torch.cat(out)


def painter(base):
    tr = Flux2Transformer2DModel.from_pretrained(base, subfolder='transformer', torch_dtype=dt)
    tr.context_embedder = nn.Identity()  # the conditioning is already in the painter's space
    return tr


def rdm_tensors(keep):
    """RDM's tensors under diffusers names, for the names keep() accepts."""
    with safe_open(RDM(), 'pt') as f:
        for k in f.keys():
            full = None
            for dk, sl in pieces(k):
                if not keep(dk):
                    continue
                full = f.get_tensor(k) if full is None else full
                if sl == 'swap':
                    h = full.shape[0] // 2
                    yield dk, torch.cat([full[h:], full[:h]])
                elif sl is not None:
                    yield dk, full[sl[0]:sl[1]]
                else:
                    yield dk, full


def teacher():
    """RDM on klein: klein's transformer with every tensor RDM has (but the context embedder, folded into the map)."""
    tr = painter(KLEIN())
    params = dict(tr.named_parameters())
    for dk, w in rdm_tensors(lambda k: k != 'context_embedder.weight'):
        params[dk].data = w.to(dt)
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
    tr = painter(TERN())
    params = dict(tr.named_parameters())
    # RDM's change to the time and modulation weights (added to the ternary painter's), as schedule '1r' has it
    with safe_open(KLEIN() + 'transformer/diffusion_pytorch_model.safetensors', 'pt') as fk:
        for dk, w in rdm_tensors(lambda k: k.startswith(MOD)):
            params[dk].data = (params[dk].data.float() + w.float() - fk.get_tensor(dk).float()).to(dt)
    for p in tr.parameters():
        p.requires_grad_(False)
    init = torch.load(f'{D}/init.pt')
    branches = {}
    for name, ab in init.items():
        parent, attr = name.rsplit('.', 1)
        br = Branch(tr.get_submodule(name), ab['A'][:rank].float().clone(), ab['B'][:, :rank].float().clone())
        setattr(tr.get_submodule(parent), attr, br)
        branches[name] = br
    offsets = {}
    for name, n in (('double_stream_modulation_img', 6 * 3072), ('double_stream_modulation_txt', 6 * 3072),
                    ('single_stream_modulation', 3 * 3072)):
        offsets[name] = Offset(getattr(tr, name), n)
        setattr(tr, name, offsets[name])
    offsets['norm_out.linear'] = Offset(tr.norm_out.linear, 2 * 3072)
    tr.norm_out.linear = offsets['norm_out.linear']
    if state:
        for name, br in branches.items():
            br.A.data, br.B.data = state['branches'][name]['A'], state['branches'][name]['B']
        for name, o in offsets.items():
            o.d.data = state['offsets'][name]
    return tr.to(dev), branches, offsets


def base_pipe():
    pipe = Flux2KleinPipeline.from_pretrained(TERN(), text_encoder=None, transformer=None, torch_dtype=dt)
    pipe.vae.to(dev)
    return pipe


def noise(seeds, pipe):
    z = torch.cat([torch.randn((1, 128, 32, 32), generator=torch.Generator('cpu').manual_seed(s)) for s in seeds]).to(dev)
    return pipe._pack_latents(z), pipe._prepare_latent_ids(z).to(dev)


def x0(tr, z, ids, ctx, pipe):
    ctx = ctx.to(dev, dt)
    v = tr(hidden_states=z.to(dt), timestep=torch.ones(z.shape[0], device=dev, dtype=dt), guidance=None,
           encoder_hidden_states=ctx, txt_ids=pipe._prepare_text_ids(ctx).to(dev), img_ids=ids, return_dict=False)[0]
    return z - v.float()


def decode(pipe, x):
    lat = x.permute(0, 2, 1).reshape(1, 128, 32, 32)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, torch.float32)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, torch.float32)
    with torch.no_grad():
        img = pipe.vae.decode(pipe._unpatchify_latents(lat * std + mean).to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type='pil')[0]


def taef2():
    """TAEF2's decoder in PyTorch (taesd.py's FLUX.2 variant): the unpatchified, normalised latent -> RGB 0..1."""
    from safetensors.torch import load_file
    from taesd import Decoder
    sd = load_file(snap('madebyollin/taef2', 'taef2.safetensors'))
    dec = Decoder(32, use_midblock_gn=True)
    dec.load_state_dict({re.sub(r'^decoder\.layers\.(\d+)', lambda m: str(int(m.group(1)) + 1), k): v
                         for k, v in sd.items() if k.startswith('decoder.')})
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec.to(dev).eval()


# ---- the steps
@torch.no_grad()
def targets(n):
    prompts = [line.strip() for line in open(f'{SCRIPTS}/../data/prompts/adapter_train.txt') if line.strip()][:n]
    held = [line.strip() for line in open(f'{SCRIPTS}/../data/calibration_prompts.txt') if line.strip()]
    t0 = time.time()
    ctx = conditioning(prompts + held)
    print(f'conditioning: {time.time() - t0:.0f}s', flush=True)
    pipe = base_pipe()
    tr = teacher()
    xs = []
    for i in range(0, len(ctx), 4):
        j = min(len(ctx), i + 4)
        z, ids = noise(range(1000 + i, 1000 + j), pipe)
        xs.append(x0(tr, z, ids, ctx[i:j], pipe).half().cpu())
        if i % 200 == 0:
            print(f'teacher {i}/{len(ctx)} {time.time() - t0:.0f}s', flush=True)
    xs = torch.cat(xs)
    torch.save(dict(prompts=prompts, ctx=ctx[:n], x0=xs[:n]), f'{D}/train.pt')
    torch.save(dict(prompts=held, ctx=ctx[n:], x0=xs[n:]), f'{D}/held.pt')
    for i in range(len(held)):
        decode(pipe, xs[n + i:n + i + 1].float().to(dev)).save(f'renders/distill/h{i}_teacher.png')
    print('saved', len(prompts), 'training prompts and', len(held), 'held out', f'{time.time() - t0:.0f}s', flush=True)


def train(steps, rank, batch=1, lr=5e-5, mse_weight=0.05):
    import lpips
    tr_data = torch.load(f'{D}/train.pt')
    held = torch.load(f'{D}/held.pt')
    n = len(tr_data['prompts'])
    pipe = base_pipe()
    path = f'{D}/student_r{rank}.pt'
    state = torch.load(path) if os.path.exists(path) else None
    tr, branches, offsets = student(rank, state)
    tr.enable_gradient_checkpointing()
    dec = taef2()
    percept = lpips.LPIPS(net='vgg', verbose=False).to(dev).eval()
    for p in percept.parameters():
        p.requires_grad_(False)

    def pixels(x):  # packed latent -> image in [-1, 1]
        return dec(pipe._unpatchify_latents(x.permute(0, 2, 1).reshape(-1, 128, 32, 32))).clamp(0, 1) * 2 - 1

    opt = torch.optim.AdamW([{'params': [p for b in branches.values() for p in (b.A, b.B)], 'lr': lr},
                             {'params': [o.d for o in offsets.values()], 'lr': 2 * lr}], weight_decay=0.0)
    if state and 'opt' in state:
        opt.load_state_dict(state['opt'])
    start = state['step'] if state else 0

    def held_eval(label=None):
        tr.eval()
        err = []
        with torch.no_grad():
            for i in range(len(held['prompts'])):
                z, ids = noise([1000 + n + i], pipe)
                xs = x0(tr, z, ids, held['ctx'][i:i + 1], pipe)
                xt = held['x0'][i:i + 1].float().to(dev)
                err.append(percept(pixels(xs), pixels(xt)).mean().item())
                if label:
                    decode(pipe, xs).save(f'renders/distill/h{i}_r{rank}_{label}.png')
        tr.train()
        return sum(err) / len(err)

    def save(s):
        torch.save(dict(step=s, rank=rank, opt=opt.state_dict(),
                        branches={k: dict(A=b.A.data.cpu(), B=b.B.data.cpu()) for k, b in branches.items()},
                        offsets={k: o.d.data.cpu() for k, o in offsets.items()}), path)

    if not state:
        print(f'held-out LPIPS before training: {held_eval("start"):.4f}', flush=True)
    tr.train()
    t0, log = time.time(), []
    for s in range(start, start + steps):
        # a fresh order every pass over the prompts
        order = torch.randperm(n, generator=torch.Generator().manual_seed((s * batch) // n))
        idx = [int(order[(s * batch + k) % n]) for k in range(batch)]
        z, ids = noise([1000 + i for i in idx], pipe)
        xs = x0(tr, z, ids, tr_data['ctx'][idx], pipe)
        xt = tr_data['x0'][idx].float().to(dev)
        with torch.no_grad():
            target = pixels(xt)
        loss = percept(pixels(xs), target).mean() + mse_weight * ((xs - xt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log.append(loss.item())
        if (s + 1) % 10 == 0:
            print(f'step {s + 1}: loss {sum(log[-10:]) / 10:.4f}, {(time.time() - t0) / (s + 1 - start):.2f}s/step',
                  flush=True)
        if (s + 1) % 250 == 0 or s + 1 == start + steps:
            save(s + 1)
            paint = (s + 1) % 500 == 0 or s + 1 == start + steps
            print(f'step {s + 1}: held-out LPIPS {held_eval(f"s{s + 1}" if paint else None):.4f}', flush=True)


def export(rank):
    """The trained branch as a LoRA file for pack_model.py --rdm (and the offsets for export_painter_schedules.py)."""
    from safetensors.torch import save_file
    st = torch.load(f'{D}/student_r{rank}.pt')
    out = {}
    for name, ab in st['branches'].items():
        out[f'transformer.{name}.lora_A.weight'] = ab['A'].to(torch.bfloat16).contiguous()
        out[f'transformer.{name}.lora_B.weight'] = ab['B'].to(torch.bfloat16).contiguous()
    save_file(out, f'{D}/rdm_trained_r{rank}.safetensors',
              metadata={'rank': str(rank), 'alpha': str(rank), 'step': str(st['step'])})
    print(f'wrote {D}/rdm_trained_r{rank}.safetensors (step {st["step"]}); then:')
    print(f'  RDM_STUDENT={D}/student_r{rank}.pt python scripts/export_painter_schedules.py')
    print(f'  python scripts/pack_model.py ... --rdm {D}/rdm_trained_r{rank}.safetensors --rdm-rank {rank}')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'prepare':
        prepare()
    elif cmd == 'targets':
        targets(int(a[0]))
    elif cmd == 'train':
        train(int(a[0]), int(a[1]) if len(a) > 1 else 32, int(a[2]) if len(a) > 2 else 1,
              float(a[3]) if len(a) > 3 else 5e-5)
    elif cmd == 'export':
        export(int(a[0]) if a else 32)
