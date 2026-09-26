"""How shallow can the reader be? Fit the conditioning map from Ternary Bonsai 1.7B's states at other layers.

The painter conditions on Qwen3-4B's layers 9/18/27; mindview replaces that with a linear map from the 1.7B's states
(adapter.py: taps 7/14/21, so the reader runs 21 of its 28 layers). Every layer the map does not need is a layer the
one-file model does not carry (about 11 MB each as trits). This fits the map for several tap sets in one pass and
scores each where it matters: in the DiT's own input space, after the context embedder (the fused map of
pack_model.py), per-token cosine against the stock encoder on held-out prompts. Each is scored as float, and ternary
(TWN codes, then each output row's group scales re-solved by least squares on the data covariance).

    python scripts/adapter_layers.py stats          # one pass of the 1.7B over the prompts, all tap sets
    python scripts/adapter_layers.py fit            # fit, ternarise and score every tap set
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
D = 'data/adapter'
CONFIGS = [(7, 14, 21), (6, 12, 18), (5, 10, 15), (7, 11, 15), (4, 8, 12), (6, 9, 12), (3, 6, 9)]
TRAIN, VAL = 1350, 150  # prompts: fit on the first, score on the rest (and on the 12 calibration prompts)
N_PAD_SAMPLES, PAD_WINDOW, PREFIX = 32, 160, 3  # as adapter.py (the same rows, so the same stored targets)
dev = 'mps'
IMG = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
READER = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-unpacked/snapshots/*/')[0]
tok = AutoTokenizer.from_pretrained(IMG + 'tokenizer')


def chat(p):
    return tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def positions(i, n_real):
    rng = np.random.default_rng(i)
    hi = min(512, n_real + PAD_WINDOW)
    pads = rng.choice(np.arange(n_real, hi), size=min(N_PAD_SAMPLES, hi - n_real), replace=False)
    return np.concatenate([np.arange(PREFIX, n_real), np.sort(pads)])


def name(cfg):
    return 'layers_' + '_'.join(map(str, cfg))


@torch.no_grad()
def stats():
    model = AutoModelForCausalLM.from_pretrained(READER, dtype=torch.bfloat16).to(dev).eval()
    prompts = [l.strip() for l in open('data/prompts/adapter_train.txt') if l.strip()][:TRAIN + VAL]
    held = [l.strip() for l in open('data/calibration_prompts.txt') if l.strip()]
    idx = json.load(open(f'{D}/index.json'))
    Y = np.load(f'{D}/targets.npy', mmap_mode='r')
    dx = 2048 * 3
    acc = {c: dict(XtX=torch.zeros(dx + 1, dx + 1, dtype=torch.float64),
                   XtY=torch.zeros(dx + 1, 7680, dtype=torch.float64), n=0, val=[]) for c in CONFIGS}
    gpu = {c: [torch.zeros(dx + 1, dx + 1, device=dev), torch.zeros(dx + 1, 7680, device=dev)] for c in CONFIGS}
    val_y, t0, BS, pending = [], time.time(), 8, 0

    def flush():
        for c in CONFIGS:
            acc[c]['XtX'] += gpu[c][0].cpu().double()
            acc[c]['XtY'] += gpu[c][1].cpu().double()
            gpu[c][0].zero_()
            gpu[c][1].zero_()

    for b in range(0, len(prompts), BS):
        batch = prompts[b:b + BS]
        inp = tok([chat(p) for p in batch], return_tensors='pt', padding='max_length', truncation=True,
                  max_length=512).to(dev)
        L = min(512, int(inp.attention_mask.sum(1).max()) + PAD_WINDOW)
        inp = {k: v[:, :L] for k, v in inp.items()}
        hs = model(**inp, output_hidden_states=True, use_cache=False).hidden_states
        nr = inp['attention_mask'].sum(1).tolist()
        for j, n in enumerate(nr):
            i = b + j
            e = idx['index'][i]
            assert e['n_real'] == n, (e, n)
            pos = torch.from_numpy(positions(i, n)).to(dev)
            y = torch.from_numpy(np.asarray(Y[e['start']:e['start'] + e['count']], dtype=np.float32))
            for c in CONFIGS:
                x = torch.cat([hs[k][j, pos].float() for k in c] + [torch.ones(len(pos), 1, device=dev)], 1)
                if i < TRAIN:
                    gpu[c][0] += x.T @ x
                    gpu[c][1] += x.T @ y.to(dev)
                    acc[c]['n'] += len(pos)
                else:
                    acc[c]['val'].append(x[:, :-1].half().cpu())
            if i >= TRAIN:
                val_y.append(y.half())
        pending += 1
        if pending >= 8:
            flush()
            pending = 0
        if b % 200 == 0:
            print(f'{b}/{len(prompts)} prompts, {time.time() - t0:.0f}s', flush=True)
    flush()
    # the 12 calibration prompts, whole
    inp = tok([chat(p) for p in held], return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(dev)
    hs = model(**inp, output_hidden_states=True, use_cache=False).hidden_states
    for c in CONFIGS:
        torch.save(dict(XtX=acc[c]['XtX'], XtY=acc[c]['XtY'], n=acc[c]['n'], val=torch.cat(acc[c]['val']),
                        held=torch.cat([hs[k] for k in c], -1).float().cpu(),
                        held_n=inp['attention_mask'].sum(1).tolist()), f'{D}/{name(c)}.pt')
    torch.save(torch.cat(val_y), f'{D}/layers_val_y.pt')
    print('done', time.time() - t0, 's')


def ternarise(Wl, Cxx, Cxy, delta=0.7):
    """Wl [out, in] -> TWN codes per group of 128 inputs; each row's group scales then re-solved by least squares
    against the data (Cxx [in, in], Cxy [in, out]: centred covariances)."""
    out, din = Wl.shape
    G = din // 128
    g = Wl.reshape(out, G, 128)
    t = torch.sign(g) * (g.abs() > delta * g.abs().mean(-1, keepdim=True))
    s_twn = (g.abs() * (t != 0)).sum(-1) / (t != 0).sum(-1).clamp_min(1)  # [out, G]
    # least squares per row: Czz[j] = T_j^T Cxx T_j (G x G), Czy[j] = T_j^T Cxy[:, j]
    U = Cxx.reshape(G, 128, G, 128).float().to(dev)
    s_ls = torch.empty(out, G, dtype=torch.float64)
    Cxy_g = Cxy.T.reshape(out, G, 128).float().to(dev)
    tt = t.float().to(dev)
    for j0 in range(0, out, 64):
        T = tt[j0:j0 + 64]  # [J, G, 128]
        tmp = torch.einsum('jgk,gkhl->jghl', T, U)
        Czz = torch.einsum('jghl,jhl->jgh', tmp, T).cpu().double()
        Czy = (T * Cxy_g[j0:j0 + 64]).sum(-1).cpu().double()
        Czz += torch.eye(G, dtype=torch.float64) * Czz.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True)[..., None] * 1e-6
        s_ls[j0:j0 + 64] = torch.linalg.solve(Czz, Czy[..., None])[..., 0]
    return t, s_twn.double(), s_ls


def fit():
    C = None
    Q = torch.load(f'{D}/qwen_extra.pt')
    val_y = torch.load(f'{D}/layers_val_y.pt').double()
    import sys as _s
    _s.path.insert(0, 'scripts')
    import pack_model as pm
    man = json.load(open(pm.PAINTER + '/manifest.json'))
    e = man['tensors']['context_embedder.weight']
    C = torch.from_numpy(np.fromfile(pm.PAINTER + '/' + e['file'], dtype=np.float16, count=e['bytes'] // 2,
                                     offset=e['offset']).reshape(3072, 7680).astype(np.float64))
    yv = val_y @ C.T  # the validation targets in the DiT's input space
    held = [(Q['held'][k, PREFIX:Q['held_n'][k]].double() @ C.T) for k in range(len(Q['held_n']))]
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
    results = {}
    for c in CONFIGS:
        S = torch.load(f'{D}/{name(c)}.pt')
        n, XtX, XtY = S['n'], S['XtX'], S['XtY']
        dx = XtX.shape[0] - 1
        mx, my = XtX[:dx, dx] / n, XtY[dx] / n
        Cxx = XtX[:dx, :dx] / n - torch.outer(mx, mx)
        Cxy = (XtY[:dx] / n - torch.outer(mx, my)) @ C.T  # [dx, 3072]: straight into the DiT's input space
        myf = my @ C.T
        reg = 1e-2 * torch.diag(Cxx).clamp_min(1e-6)
        W = torch.linalg.solve(Cxx + torch.diag(reg), Cxy)  # [dx, 3072]
        b = myf - mx @ W
        t, s_twn, s_ls = ternarise(W.T.contiguous(), Cxx, Cxy)
        G = dx // 128

        def wt(s):
            return (t.double() * s[..., None]).reshape(3072, dx).T  # [dx, 3072]

        row = {}
        for label, Wx in [('float', W), ('ternary_twn', wt(s_twn)), ('ternary_ls', wt(s_ls))]:
            bx = myf - mx @ Wx
            xv = S['val'].double()
            row[label] = dict(
                val=cos(xv @ Wx + bx, yv),
                held=float(np.mean([cos(S['held'][k, PREFIX:S['held_n'][k]].double() @ Wx + bx, held[k])
                                    for k in range(len(held))])))
        row['zeros'] = (t == 0).float().mean().item()
        row['reader_layers'] = max(c)
        results[name(c)] = row
        torch.save(dict(W=W.float(), b=b.float(), t=t.to(torch.int8), s_ls=s_ls.float(), s_twn=s_twn.float(),
                        b_ls=(myf - mx @ wt(s_ls)).float(), taps=c), f'{D}/fused_{name(c)}.pt')
        print(name(c), json.dumps(row), flush=True)
    json.dump(results, open(f'{D}/layers_results.json', 'w'), indent=1)


def final():
    """The map to ship, for one tap set: fitted on every prompt (the training rows and the validation rows), float
    and ternary; scored on the 12 calibration prompts (never fitted on)."""
    c = tuple(int(x) for x in sys.argv[2:5])
    sys.path.insert(0, 'scripts')
    import pack_model as pm
    man = json.load(open(pm.PAINTER + '/manifest.json'))
    e = man['tensors']['context_embedder.weight']
    C = torch.from_numpy(np.fromfile(pm.PAINTER + '/' + e['file'], dtype=np.float16, count=e['bytes'] // 2,
                                     offset=e['offset']).reshape(3072, 7680).astype(np.float64))
    S = torch.load(f'{D}/{name(c)}.pt')
    xv = S['val'].double()
    xv = torch.cat([xv, torch.ones(len(xv), 1, dtype=torch.float64)], 1)
    yv = torch.load(f'{D}/layers_val_y.pt').double()
    n = S['n'] + len(xv)
    XtX, XtY = S['XtX'] + xv.T @ xv, S['XtY'] + xv.T @ yv
    dx = XtX.shape[0] - 1
    mx, my = XtX[:dx, dx] / n, XtY[dx] / n
    Cxx = XtX[:dx, :dx] / n - torch.outer(mx, mx)
    Cxy = (XtY[:dx] / n - torch.outer(mx, my)) @ C.T
    myf = my @ C.T
    W = torch.linalg.solve(Cxx + torch.diag(1e-2 * torch.diag(Cxx).clamp_min(1e-6)), Cxy)
    b = myf - mx @ W
    t, s_twn, s_ls = ternarise(W.T.contiguous(), Cxx, Cxy)
    Wt = (t.double() * s_ls[..., None]).reshape(3072, dx).T
    b_ls = myf - mx @ Wt
    Q = torch.load(f'{D}/qwen_extra.pt')
    held = [(Q['held'][k, PREFIX:Q['held_n'][k]].double() @ C.T) for k in range(len(Q['held_n']))]
    cos = lambda a, b_: torch.nn.functional.cosine_similarity(a, b_, dim=-1).mean().item()
    for label, Wx, bx in [('float', W, b), ('ternary_ls', Wt, b_ls)]:
        h = np.mean([cos(S['held'][k, PREFIX:S['held_n'][k]].double() @ Wx + bx, held[k]) for k in range(len(held))])
        print(f'{name(c)} final {label}: held-out cosine {h:.4f}')
    torch.save(dict(W=W.float(), b=b.float(), t=t.to(torch.int8), s_ls=s_ls.float(), b_ls=b_ls.float(), taps=c),
               f'{D}/fused_final_{name(c)}.pt')


if __name__ == '__main__':
    {'stats': stats, 'fit': fit, 'final': final}[sys.argv[1]]()
