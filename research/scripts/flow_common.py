"""Shared: load Ternary Bonsai 1.7B (unpacked) and capture everything one prompt does, layer by layer."""
import glob, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = os.environ.get('PROMPT', 'a bonsai tree made of glowing circuitry in a dark museum, volumetric light')
DEV = 'mps'
torch.set_grad_enabled(False)


def load():
    path = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-unpacked/snapshots/*/')[0]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, attn_implementation='eager').to(DEV).eval()
    return tok, model


def capture(tok, model, prompt=PROMPT):
    """Per layer l: resid[l] (layer input), attn[l] (attention's addition), mid[l], mlp[l] (MLP's addition),
    act[l] (the 6144 neuron values fed to down_proj), v[l] (value vectors), probs[l] (heads x n x n)."""
    text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors='pt').input_ids.to(DEV)
    words = [tok.decode([i]) for i in ids[0]]
    rec = {'resid': {}, 'attn': {}, 'act': {}, 'v': {}, 'out': {}}
    hooks = []
    f = lambda x: x[0].float().cpu()
    for i, L in enumerate(model.model.layers):
        hooks += [
            L.register_forward_pre_hook(lambda m, a, i=i: rec['resid'].__setitem__(i, f(a[0]))),
            L.register_forward_hook(lambda m, a, o, i=i: rec['out'].__setitem__(i, f(o[0] if isinstance(o, tuple) else o))),
            L.self_attn.register_forward_hook(lambda m, a, o, i=i: rec['attn'].__setitem__(i, f(o[0]))),
            L.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: rec['act'].__setitem__(i, f(a[0]))),
            L.self_attn.v_proj.register_forward_hook(lambda m, a, o, i=i: rec['v'].__setitem__(i, f(o))),
        ]
    out = model(input_ids=ids, output_attentions=True)
    for h in hooks:
        h.remove()
    nl = len(model.model.layers)
    resid = torch.stack([rec['resid'][i] for i in range(nl)] + [rec['out'][nl - 1]])
    attn = torch.stack([rec['attn'][i] for i in range(nl)])
    mid = resid[:-1] + attn
    return dict(
        ids=ids[0].cpu(), words=words, resid=resid, attn=attn, mid=mid, mlp=resid[1:] - mid,
        act=torch.stack([rec['act'][i] for i in range(nl)]), v=torch.stack([rec['v'][i] for i in range(nl)]),
        probs=torch.stack([a[0].float().cpu() for a in out.attentions]),
        end=next(i for i, w in enumerate(words) if i > 3 and w == '<|im_end|>'),
    )


def logit_lens(model, H, chunk=2048):
    """argmax token and its probability for each row of H (points in the residual stream)."""
    ids, conf = [], []
    for c in H.split(chunk):
        x = model.model.norm(c.float().half().to(DEV))
        p = torch.softmax(model.lm_head(x).float(), -1)
        v, i = p.max(-1)
        ids.append(i.cpu()); conf.append(v.cpu())
    return torch.cat(ids), torch.cat(conf)
