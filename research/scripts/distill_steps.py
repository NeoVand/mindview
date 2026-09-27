"""Train side branches for Fast (2 steps, the first at 256 x 256) and Best (4 steps) on the ternary painter.

Like distill_1step.py for Instant, but a step at a time along the teacher's own path:
    fast   teacher: FLUX.2 klein 4B at full precision with the 2-step LoRA (radames/FLUX.2-klein-Sana-Sprint), run as the
           browser runs Fast: step 1 at 16 x 16 patches from sigma 1 with 128 text rows; its clean-picture guess
           upsampled in latent space and noised back to sigma 0.885 with fresh noise; step 2 at 32 x 32 with 256 rows.
           Student: the ternary painter + the 2-step LoRA as a side branch (rank 8, from its SVD) widened to rank R.
    best   teacher: klein at full precision, 4 steps (the schedule the browser uses), with the prompt's own text rows
           (prompt + at least 8 pads, rounded up to 64, as Painter.textLength 'auto'). Student: the ternary painter +
           a new rank-R branch (starting at zero) - the ternary rounding is what it learns to undo.
Both students also learn an offset on the modulation of each step (it is precomputed per step in the browser).
Loss, for each step at the teacher's input to that step: LPIPS between the two clean-picture guesses decoded by TAEF2,
plus a little latent mean squared error. The held-out prompts are painted by the student's own path, end to end.

    python scripts/distill_steps.py targets MODE N       # teacher paths for N prompts (+ the 12 held out)
    python scripts/distill_steps.py train MODE STEPS [R] [B] [LR]
    python scripts/distill_steps.py export MODE [R]      # the branch as a LoRA file; offsets for the schedules
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_1step as d1  # noqa: E402  (chdir to WORK; the shared pieces)

D, dev, dt = d1.D, d1.dev, d1.dt
S1 = 0.8854396343231201
SIG4 = [1.0, 0.9580853581428528, 0.8839818835258484, 0.7174965739250183, 0.0]
SANA = lambda: d1.snap('radames/FLUX.2-klein-Sana-Sprint', 'pytorch_lora_weights.safetensors')  # noqa: E731
MODES = {
    'fast': dict(steps=[dict(side=16, sigma=1.0, rows=128), dict(side=32, sigma=S1, rows=256)], schedule='2'),
    'best': dict(steps=[dict(side=32, sigma=SIG4[s], rows='auto') for s in range(4)], schedule='4'),
}
MODULES = [('double_stream_modulation_img', 6 * 3072), ('double_stream_modulation_txt', 6 * 3072),
           ('single_stream_modulation', 3 * 3072), ('norm_out.linear', 2 * 3072)]


def auto_rows(n_real):
    return max(64, -(-(n_real + 8) // 64) * 64)


def ids_for(pipe, side, b):
    return pipe._prepare_latent_ids(torch.zeros(b, 128, side, side)).to(dev)


def noise(seeds, side, pipe):
    z = torch.cat([torch.randn((1, 128, side, side), generator=torch.Generator('cpu').manual_seed(int(s))) for s in seeds])
    return pipe._pack_latents(z.to(dev))


def velocity(tr, x, side, sigma, ctx, pipe):
    ctx = ctx.to(dev, dt)
    return tr(hidden_states=x.to(dt), timestep=torch.full((x.shape[0],), sigma, device=dev, dtype=dt), guidance=None,
              encoder_hidden_states=ctx, txt_ids=pipe._prepare_text_ids(ctx).to(dev),
              img_ids=ids_for(pipe, side, x.shape[0]), return_dict=False)[0].float()


def upsample(x0, pipe):
    """A 16 x 16-patch guess -> 32 x 32 patches: bilinear in the unpatchified latent (as Painter.upsampleLatent)."""
    lat = pipe._unpatchify_latents(x0.permute(0, 2, 1).reshape(-1, 128, 16, 16))
    lat = F.interpolate(lat, scale_factor=2, mode='bilinear', align_corners=False)
    return pipe._pack_latents(pipe._patchify_latents(lat))


def seeds_of(mode, i):
    return (5000 + 2 * i, 5000 + 2 * i + 1) if mode == 'fast' else (5000 + i,)


def rows_of(st, n_real):
    return auto_rows(n_real) if st['rows'] == 'auto' else st['rows']


def path(tr, mode, seeds, ctx, n_real, pipe, keep=False):
    """The whole painting for one prompt with a painter tr: [(input, clean guess) per step] if keep, else the picture."""
    out, x = [], None
    steps = MODES[mode]['steps']
    for s, st in enumerate(steps):
        if mode == 'fast':
            x = noise([seeds[0]], 16, pipe) if s == 0 else (1 - S1) * upsample(x0, pipe) + S1 * noise([seeds[1]], 32, pipe)
        elif s == 0:
            x = noise([seeds[0]], 32, pipe)
        CUR['step'] = s
        v = velocity(tr, x, st['side'], st['sigma'], ctx[:, :rows_of(st, n_real)], pipe)
        x0 = x - st['sigma'] * v
        if keep:
            out.append((x.half().cpu(), x0.half().cpu()))
        if mode == 'best' and s < 3:
            x = x + (SIG4[s + 1] - SIG4[s]) * v
    return out if keep else x0


CUR = {'step': 0}


class StepOffset(nn.Module):
    """A modulation module's output plus a learned constant for the current step."""

    def __init__(self, mod, n, steps):
        super().__init__()
        self.mod, self.d = mod, nn.Parameter(torch.zeros(steps, n))

    def forward(self, *a, **kw):
        return self.mod(*a, **kw) + self.d[CUR['step']].to(dt)


def teacher(mode):
    tr = d1.painter(d1.KLEIN())
    if mode == 'fast':
        with safe_open(SANA(), 'pt') as s:
            for m in sorted({k.rsplit('.lora_', 1)[0] for k in s.keys()}):
                lin = tr.get_submodule(m.removeprefix('transformer.'))
                lin.weight.data = (lin.weight.data.float() + s.get_tensor(f'{m}.lora_B.weight').float()
                                   @ s.get_tensor(f'{m}.lora_A.weight').float()).to(dt)
    return tr.to(dev).eval()


def modules():
    with safe_open(SANA(), 'pt') as s:
        return sorted({k.rsplit('.lora_', 1)[0].removeprefix('transformer.') for k in s.keys()})


def student(mode, rank, state=None):
    tr = d1.painter(d1.TERN())
    for p in tr.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(0)
    branches = {}
    sana = safe_open(SANA(), 'pt') if mode == 'fast' else None
    for name in modules():
        lin = tr.get_submodule(name)
        out_f, in_f = lin.weight.shape
        A = torch.randn(rank, in_f, generator=g) / in_f ** 0.5
        B = torch.zeros(out_f, rank)
        if sana is not None:  # the 2-step LoRA's best rank-8 approximation, then room for more
            BA = (sana.get_tensor(f'transformer.{name}.lora_B.weight').float().to(dev)
                  @ sana.get_tensor(f'transformer.{name}.lora_A.weight').float().to(dev))
            U, S, V = torch.svd_lowrank(BA, q=16, niter=4)
            sq = S[:8].sqrt()
            A[:8], B[:, :8] = (sq[:, None] * V[:, :8].T).cpu(), (U[:, :8] * sq[None, :]).cpu()
        parent, attr = name.rsplit('.', 1)
        br = d1.Branch(lin, A, B)
        setattr(tr.get_submodule(parent), attr, br)
        branches[name] = br
    n_steps = len(MODES[mode]['steps'])
    offsets = {}
    for name, n in MODULES:
        parent, attr = ('norm_out', 'linear') if name == 'norm_out.linear' else ('', name)
        host = tr.get_submodule(parent) if parent else tr
        offsets[name] = StepOffset(getattr(host, attr), n, n_steps)
        setattr(host, attr, offsets[name])
    if state:
        for name, br in branches.items():
            br.A.data, br.B.data = state['branches'][name]['A'], state['branches'][name]['B']
        for name, o in offsets.items():
            o.d.data = state['offsets'][name]
    return tr.to(dev), branches, offsets


# ---- the steps
@torch.no_grad()
def targets(mode, n):
    from transformers import AutoTokenizer
    prompts = [line.strip() for line in open(f'{d1.SCRIPTS}/../data/prompts/adapter_train.txt') if line.strip()][:n]
    held = [line.strip() for line in open(f'{d1.SCRIPTS}/../data/calibration_prompts.txt') if line.strip()]
    tok = AutoTokenizer.from_pretrained(d1.TERN() + 'tokenizer')
    n_real = [len(tok(tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)).input_ids) for p in prompts + held]
    t0 = time.time()
    ctx = d1.conditioning(prompts + held)
    pipe = d1.base_pipe()
    tr = teacher(mode)
    steps = len(MODES[mode]['steps'])
    xs, x0s = [[] for _ in range(steps)], [[] for _ in range(steps)]
    for i in range(len(ctx)):
        for s, (x, x0) in enumerate(path(tr, mode, seeds_of(mode, i), ctx[i:i + 1], n_real[i], pipe, keep=True)):
            xs[s].append(x)
            x0s[s].append(x0)
        if i % 400 == 0:
            print(f'teacher {i}/{len(ctx)} {time.time() - t0:.0f}s', flush=True)
    xs, x0s = [torch.cat(v) for v in xs], [torch.cat(v) for v in x0s]
    torch.save(dict(prompts=prompts, ctx=ctx[:n], n_real=n_real[:n], x=[v[:n] for v in xs], x0=[v[:n] for v in x0s]),
               f'{D}/{mode}_train.pt')
    torch.save(dict(prompts=held, ctx=ctx[n:], n_real=n_real[n:], first=n, x0=[v[n:] for v in x0s]), f'{D}/{mode}_held.pt')
    for j in range(len(held)):
        d1.decode(pipe, x0s[-1][n + j:n + j + 1].float().to(dev)).save(f'renders/distill/{mode}_h{j}_teacher.png')
    print('saved', n, 'training prompts and', len(held), 'held out', f'{time.time() - t0:.0f}s', flush=True)


def train(mode, steps, rank, batch=4, lr=1e-4, mse_weight=0.05):
    import lpips
    data = torch.load(f'{D}/{mode}_train.pt')
    held = torch.load(f'{D}/{mode}_held.pt')
    n, cfg = len(data['prompts']), MODES[mode]['steps']
    pipe = d1.base_pipe()
    ck = f'{D}/{mode}_r{rank}.pt'
    state = torch.load(ck, map_location='cpu', weights_only=False) if os.path.exists(ck) else None
    tr, branches, offsets = student(mode, rank, state)
    tr.enable_gradient_checkpointing()
    dec = d1.taef2()
    percept = lpips.LPIPS(net='vgg', verbose=False).to(dev).eval()
    for p in percept.parameters():
        p.requires_grad_(False)

    def pixels(x, side):
        return dec(pipe._unpatchify_latents(x.permute(0, 2, 1).reshape(-1, 128, side, side))).clamp(0, 1) * 2 - 1

    opt = torch.optim.AdamW([{'params': [p for b in branches.values() for p in (b.A, b.B)], 'lr': lr},
                             {'params': [o.d for o in offsets.values()], 'lr': 2 * lr}], weight_decay=0.0)
    if state and 'opt' in state:
        opt.load_state_dict(state['opt'])
    start = state['step'] if state else 0
    # prompts grouped by their text rows (Best uses the prompt's own rows, so a batch shares them)
    rows = [rows_of(cfg[-1], r) for r in data['n_real']]
    buckets = {}
    for i, r in enumerate(rows):
        buckets.setdefault(r, []).append(i)
    keys = sorted(buckets)
    weights = torch.tensor([len(buckets[k]) for k in keys], dtype=torch.float)

    def held_eval(label=None):
        tr.eval()
        err = []
        with torch.no_grad():
            for j in range(len(held['prompts'])):
                x0 = path(tr, mode, seeds_of(mode, held['first'] + j), held['ctx'][j:j + 1], held['n_real'][j], pipe)
                err.append(percept(pixels(x0, 32), pixels(held['x0'][-1][j:j + 1].float().to(dev), 32)).mean().item())
                if label:
                    d1.decode(pipe, x0).save(f'renders/distill/{mode}_h{j}_r{rank}_{label}.png')
        tr.train()
        return sum(err) / len(err)

    def save(s, best, to=ck):
        torch.save(dict(step=s, rank=rank, mode=mode, opt=opt.state_dict(), best=best,
                        branches={k: dict(A=b.A.data.cpu(), B=b.B.data.cpu()) for k, b in branches.items()},
                        offsets={k: o.d.data.cpu() for k, o in offsets.items()}), to)

    best = state.get('best', 1e9) if state else held_eval('start')
    if not state:
        print(f'held-out LPIPS before training: {best:.4f}', flush=True)
    base_lrs, total = [g['lr'] for g in opt.param_groups], start + steps
    gen = torch.Generator().manual_seed(start)
    tr.train()
    t0, log = time.time(), []
    for s in range(start, start + steps):
        k = keys[int(torch.multinomial(weights, 1, generator=gen))]
        pool = buckets[k]
        idx = [pool[int(j)] for j in torch.randint(len(pool), (batch,), generator=gen)]
        step = int(torch.randint(len(cfg), (1,), generator=gen))
        st = cfg[step]
        CUR['step'] = step
        x = data['x'][step][idx].float().to(dev)
        xt = data['x0'][step][idx].float().to(dev)
        ctx = data['ctx'][idx][:, :rows_of(st, data['n_real'][idx[0]])]
        x0 = x - st['sigma'] * velocity(tr, x, st['side'], st['sigma'], ctx, pipe)
        with torch.no_grad():
            target = pixels(xt, st['side'])
        loss = percept(pixels(x0, st['side']), target).mean() + mse_weight * ((x0 - xt) ** 2).mean()
        f = float(0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (s - start) / max(1, total - start))))
        for g, b in zip(opt.param_groups, base_lrs):
            g['lr'] = b * f
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log.append(loss.item())
        if (s + 1) % 20 == 0:
            print(f'step {s + 1}: loss {sum(log[-20:]) / 20:.4f}, {(time.time() - t0) / (s + 1 - start):.2f}s/step',
                  flush=True)
        if (s + 1) % 250 == 0 or s + 1 == total:
            paint = (s + 1) % 1000 == 0 or s + 1 == total
            h = held_eval(f's{s + 1}' if paint else None)
            print(f'step {s + 1}: held-out LPIPS {h:.4f}', flush=True)
            if h < best:
                best = h
                save(s + 1, best, f'{D}/{mode}_r{rank}_best.pt')
            save(s + 1, best)


def export(mode, rank):
    from safetensors.torch import save_file
    st = torch.load(f'{D}/{mode}_r{rank}_best.pt', map_location='cpu', weights_only=False)
    out = {}
    for name, ab in st['branches'].items():
        out[f'transformer.{name}.lora_A.weight'] = ab['A'].to(torch.bfloat16).contiguous()
        out[f'transformer.{name}.lora_B.weight'] = ab['B'].to(torch.bfloat16).contiguous()
    save_file(out, f'{D}/{mode}_trained_r{rank}.safetensors',
              metadata={'rank': str(rank), 'alpha': str(rank), 'step': str(st['step'])})
    torch.save(st['offsets'], f'{D}/{mode}_offsets.pt')
    print(f'wrote {D}/{mode}_trained_r{rank}.safetensors and {D}/{mode}_offsets.pt (step {st["step"]})')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'targets':
        targets(a[0], int(a[1]))
    elif cmd == 'train':
        train(a[0], int(a[1]), int(a[2]) if len(a) > 2 else 32, int(a[3]) if len(a) > 3 else 4,
              float(a[4]) if len(a) > 4 else 1e-4)
    elif cmd == 'export':
        export(a[0], int(a[1]) if len(a) > 1 else 32)
