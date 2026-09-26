"""Linear adapter: make a ternary Bonsai LLM speak the dialect the image model expects.

Bonsai Image conditions on Qwen3-4B hidden states from layers 9/18/27 (7680-d per token, 512 padded tokens).
We learn a ridge-regression map  [bonsai taps] -> [qwen 9|18|27]  per token, from ~3000 prompts, so a ternary
LLM (Ternary-Bonsai-4B or -1.7B) can replace the full-precision encoder.

  adapter.py targets                 # pass A: run Qwen3-4B, store targets at selected positions
  adapter.py stats  <src>            # pass B: run the ternary LLM, accumulate X^T X, X^T Y  (src: bonsai4b | bonsai17)
  adapter.py fit    <src> [lambda]   # solve ridge, report held-out fit
  adapter.py apply  <src> "<prompt>" ...   # write adapted embeds into data/prompt_cache as encoder '<src>_adapt'
"""
import sys, glob, os, json, hashlib, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
SRC = {
    'bonsai4b': ('Ternary-Bonsai-4B-unpacked', (9, 18, 27)),
    'bonsai17': ('Ternary-Bonsai-1.7B-unpacked', (7, 14, 21)),
}
SRC['bonsai4bt'], SRC['bonsai17t'] = SRC['bonsai4b'], SRC['bonsai17']  # ternarized adapters (adapter_ternarize.py)
TGT_LAYERS = (9, 18, 27)
N_PAD_SAMPLES = 32
PAD_WINDOW = 160  # pads are sampled from the first PAD_WINDOW positions after the prompt; causal attention means
                  # we can stop the forward pass right after the last sampled position (4x cheaper than all 512)
PREFIX = 3  # '<|im_start|>', 'user', '\n' are identical for every prompt -> copied verbatim, not regressed
D = 'data/adapter'
os.makedirs(D, exist_ok=True)
dev = 'mps'
tok = AutoTokenizer.from_pretrained(ROOT + 'tokenizer')


def chat(p):
    return tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def load(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()


@torch.no_grad()
def taps(model, prompts, layers, full=False):
    inp = tok([chat(p) for p in prompts], return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(dev)
    if not full:
        L = min(512, int(inp.attention_mask.sum(1).max()) + PAD_WINDOW)
        inp = {k: v[:, :L] for k, v in inp.items()}
    out = model(**inp, output_hidden_states=True, use_cache=False)
    h = torch.cat([out.hidden_states[k] for k in layers], -1)  # [B, 512, taps*d]
    return h, inp['attention_mask'].sum(1).tolist()


def positions(i, n_real):
    rng = np.random.default_rng(i)
    hi = min(512, n_real + PAD_WINDOW)
    pads = rng.choice(np.arange(n_real, hi), size=min(N_PAD_SAMPLES, hi - n_real), replace=False)
    return np.concatenate([np.arange(PREFIX, n_real), np.sort(pads)])


prompts = [l.strip() for l in open('data/prompts/adapter_train.txt') if l.strip()][:1500]
held = [l.strip() for l in open('data/calibration_prompts.txt') if l.strip()]
mode = sys.argv[1]
BS = 8

if mode == 'targets':
    model = load(ROOT + 'text_encoder')
    rows, index = [], []
    t0 = time.time()
    total = 0
    Y = np.lib.format.open_memmap(f'{D}/targets.npy', mode='w+', dtype=np.float16, shape=(len(prompts) * 140, 7680))
    for b in range(0, len(prompts), BS):
        h, nr = taps(model, prompts[b:b + BS], TGT_LAYERS)
        for j, n in enumerate(nr):
            pos = positions(b + j, n)
            Y[total:total + len(pos)] = h[j, pos].float().cpu().numpy().astype(np.float16)
            index.append(dict(i=b + j, n_real=n, start=total, count=len(pos)))
            total += len(pos)
        if b % 200 == 0:
            print(f'{b}/{len(prompts)} prompts, {total} rows, {time.time()-t0:.0f}s', flush=True)
    # the constant chat-template prefix, and held-out targets for evaluation
    h, nr = taps(model, held, TGT_LAYERS, full=True)
    torch.save(dict(prefix=h[0, :PREFIX].float().cpu(), held=h.float().cpu(), held_n=nr), f'{D}/qwen_extra.pt')
    json.dump(dict(rows=total, index=index), open(f'{D}/index.json', 'w'))
    print('done', total, 'rows')

elif mode == 'stats':
    src = sys.argv[2]
    repo, layers = SRC[src]
    model = load(glob.glob(f'/Users/neo/.cache/huggingface/hub/models--prism-ml--{repo}/snapshots/*/')[0])
    idx = json.load(open(f'{D}/index.json'))
    Y = np.load(f'{D}/targets.npy', mmap_mode='r')
    dx = model.config.hidden_size * len(layers)
    XtX = torch.zeros(dx + 1, dx + 1, dtype=torch.float64)
    XtY = torch.zeros(dx + 1, 7680, dtype=torch.float64)
    YtY_diag = torch.zeros(7680, dtype=torch.float64); Ysum = torch.zeros(7680, dtype=torch.float64); n_tot = 0
    t0 = time.time()
    acc_x, acc_y = [], []
    for b in range(0, len(prompts), BS):
        h, nr = taps(model, prompts[b:b + BS], layers)
        for j, n in enumerate(nr):
            e = idx['index'][b + j]
            assert e['n_real'] == n, (e, n)  # same tokenizer + template -> same positions
            pos = positions(b + j, n)
            acc_x.append(h[j, pos].float())
            acc_y.append(torch.from_numpy(np.asarray(Y[e['start']:e['start'] + e['count']], dtype=np.float32)).to(dev))
        if len(acc_x) >= 32 or b + BS >= len(prompts):
            X = torch.cat(acc_x); X = torch.cat([X, torch.ones(len(X), 1, device=dev)], 1)
            Yb = torch.cat(acc_y)
            XtX += (X.T @ X).cpu().double(); XtY += (X.T @ Yb).cpu().double()
            YtY_diag += (Yb * Yb).sum(0).cpu().double(); Ysum += Yb.sum(0).cpu().double(); n_tot += len(X)
            acc_x, acc_y = [], []
        if b % 200 == 0:
            print(f'{b}/{len(prompts)} prompts, {time.time()-t0:.0f}s', flush=True)
    h, nr = taps(model, held, layers, full=True)
    torch.save(dict(XtX=XtX, XtY=XtY, YtY_diag=YtY_diag, Ysum=Ysum, n=n_tot, held=h.float().cpu(), held_n=nr), f'{D}/stats_{src}.pt')
    print('done', n_tot, 'rows')

elif mode == 'fit':
    src = sys.argv[2]
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2
    S = torch.load(f'{D}/stats_{src}.pt')
    n = S['n']; XtX, XtY = S['XtX'], S['XtY']
    dx = XtX.shape[0] - 1
    mx = XtX[:dx, dx] / n; my = XtY[dx] / n
    Cxx = XtX[:dx, :dx] / n - torch.outer(mx, mx)
    Cxy = XtY[:dx] / n - torch.outer(mx, my)
    reg = lam * torch.diag(Cxx).clamp_min(1e-6)
    W = torch.linalg.solve(Cxx + torch.diag(reg), Cxy)  # [dx, 7680]
    b = my - mx @ W
    torch.save(dict(W=W.float(), b=b.float()), f'{D}/adapter_{src}.pt')
    Q = torch.load(f'{D}/qwen_extra.pt')
    r2s, coss = [], []
    for k in range(len(held)):
        nq = Q['held_n'][k]
        x = S['held'][k, PREFIX:nq].double(); y = Q['held'][k, PREFIX:nq].double()
        yp = x @ W + b
        for L in range(3):
            sl = slice(L * 2560, (L + 1) * 2560)
            coss.append(torch.nn.functional.cosine_similarity(yp[:, sl], y[:, sl], dim=-1).mean().item())
        var = ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-6)
        r2s.append((1 - ((y - yp) ** 2).sum(0) / var).mean().item())
    print(f'{src} lambda={lam}: held-out per-token cosine (real tokens, avg over 3 taps) = {np.mean(coss):.3f}; mean per-dim R2 = {np.mean(r2s):.3f}')

elif mode == 'apply':
    src = sys.argv[2]
    repo, layers = SRC[src]
    A = torch.load(f'{D}/adapter_{src}.pt'); Q = torch.load(f'{D}/qwen_extra.pt')
    model = load(glob.glob(f'/Users/neo/.cache/huggingface/hub/models--prism-ml--{repo}/snapshots/*/')[0])
    for p in sys.argv[3:]:
        h, nr = taps(model, [p], layers, full=True)
        e = (h[0].float().cpu() @ A['W'] + A['b'])  # [512, 7680]
        e[:PREFIX] = Q['prefix']
        key = hashlib.sha1(f'{src}_adapt|{p}'.encode()).hexdigest()[:12]
        torch.save(dict(prompt=p, encoder=f'{src}_adapt', embeds=e[None].to(torch.bfloat16), n_real=nr[0]), f'data/prompt_cache/{key}.pt')
        print('wrote', key, p)
