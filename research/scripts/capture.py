"""Run a Bonsai model with hooks and capture every intermediate activation for each generated token.

Captured per layer: residual stream in/out, attention probs (all heads), q/k/v, attention output,
MLP gate/up/act (the 'neurons' fed to down_proj), MLP output. Plus final logits.
"""
import sys, glob, time, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = sys.argv[1] if len(sys.argv) > 1 else 'Ternary-Bonsai-1.7B-unpacked'
prompt = sys.argv[2] if len(sys.argv) > 2 else 'What does it feel like to be a thought?'
n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 24
path = glob.glob(f'/Users/neo/.cache/huggingface/hub/models--prism-ml--{repo}/snapshots/*/')[0]
dev = 'mps'
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, attn_implementation='eager').to(dev).eval()
cfg = model.config
print(cfg.model_type, cfg.num_hidden_layers, 'layers', cfg.hidden_size, 'hidden', cfg.intermediate_size, 'ffn')

rec = {}
def hook(name):
    def fn(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        rec.setdefault(name, []).append(x[0, -1].float().cpu().numpy())
        if isinstance(inp, tuple) and len(inp) and torch.is_tensor(inp[0]) and name.endswith('down_proj'):
            rec.setdefault(name + '.in', []).append(inp[0][0, -1].float().cpu().numpy())
    return fn

for i, L in enumerate(model.model.layers):
    for sub in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'self_attn', 'mlp']:
        m = L.get_submodule(sub)
        m.register_forward_hook(hook(f'L{i}.{sub}'))

msgs = [{'role': 'user', 'content': prompt}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors='pt').input_ids.to(dev)
print('prompt tokens:', ids.shape[1])

# manual greedy loop with KV cache so we can grab per-step attention + hidden states
past = None
cur = ids
steps = []
t0 = time.time()
with torch.no_grad():
    for step in range(n_new):
        rec.clear()
        out = model(input_ids=cur, past_key_values=past, use_cache=True, output_hidden_states=True, output_attentions=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        top = torch.topk(probs, 20)
        nxt = int(top.indices[0])
        hs = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()  # (L+1, d)
        att = [a[0, :, -1, :].float().cpu().numpy() for a in out.attentions]  # per layer (heads, ctx)
        steps.append(dict(token=nxt, text=tok.decode([nxt]),
                          top=[(tok.decode([int(i)]), float(p)) for i, p in zip(top.indices, top.values)],
                          hidden=hs, attn=att, rec={k: v[-1] for k, v in rec.items()},
                          entropy=float(-(probs * torch.log(probs + 1e-12)).sum())))
        cur = torch.tensor([[nxt]], device=dev)
        if nxt == tok.eos_token_id:
            break
dt = time.time() - t0
print(f'{len(steps)} tokens in {dt:.2f}s ({len(steps)/dt:.1f} tok/s, with full capture)')
print('OUTPUT:', repr(''.join(s["text"] for s in steps)))
s = steps[-1]
nbytes = s['hidden'].nbytes + sum(a.nbytes for a in s['attn']) + sum(v.nbytes for v in s['rec'].values())
print(f'captured floats per token (last step): {nbytes/4/1e6:.2f} M floats = {nbytes/1e6:.1f} MB fp32')
for k in ['L0.mlp.down_proj.in', 'L0.self_attn.q_proj', 'L0.mlp.gate_proj', 'L0.mlp']:
    print('  ', k, s['rec'][k].shape)
print('hidden norms per layer (last token):', np.round(np.linalg.norm(s['hidden'], axis=1), 1).tolist())
for st in steps[:8]:
    print(f"  {st['text']!r:14s} H={st['entropy']:.2f}  top: " + ', '.join(f'{t!r}:{p:.2f}' for t, p in st['top'][:5]))
np.savez_compressed(f'data/capture_{repo}.npz',
                    tokens=np.array([s['token'] for s in steps]),
                    hidden=np.stack([s['hidden'] for s in steps]),
                    neurons=np.stack([np.stack([s['rec'][f'L{i}.mlp.down_proj.in'] for i in range(cfg.num_hidden_layers)]) for s in steps]),
                    )
json.dump(dict(prompt=prompt, text=[s['text'] for s in steps], top=[s['top'] for s in steps], entropy=[s['entropy'] for s in steps]),
          open(f'data/capture_{repo}.json', 'w'), indent=1)
