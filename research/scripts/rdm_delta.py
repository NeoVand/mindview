"""How low-rank is a 1-step fine-tune of FLUX.2 klein 4B? (epfl-vita/flux2-klein-1step-rdm, a full fine-tune)

If W_rdm - W_klein is well approximated at low rank, it can run like the few-step LoRA: a side branch on the ternary
painter. This maps RDM's tensor names (BFL's original layout, q/k/v fused) onto diffusers' (klein and the painter),
checks every pairing by correlation, reports how much each part changed and how much of each matrix's change the
top r singular vectors hold, and saves the change of every painter matrix as a rank-R LoRA (diffusers naming), plus
the exact change of everything else.

    python scripts/rdm_delta.py [R]      # default R = 256
"""
import glob
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdm_delta_map import pieces  # noqa: E402

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
R = int(sys.argv[1]) if len(sys.argv) > 1 else 256
RDM = glob.glob('/Users/neo/.cache/huggingface/hub/models--epfl-vita--flux2-klein-1step-rdm/snapshots/*/model.safetensors')[0]
KLEIN = glob.glob('/Users/neo/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/transformer/'
                  'diffusion_pytorch_model.safetensors')[0]
OUT = 'data/rdm'
os.makedirs(OUT, exist_ok=True)


# the painter's ternary matrices (the side branch's targets)
TARGET = ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.add_q_proj', 'attn.add_k_proj', 'attn.add_v_proj',
          'attn.to_out.0', 'attn.to_add_out', 'ff.linear_in', 'ff.linear_out', 'ff_context.linear_in',
          'ff_context.linear_out', 'attn.to_qkv_mlp_proj', 'attn.to_out.weight')
RANKS = [8, 32, 64, 128, 256, 512]
lora, rest, report = {}, {}, {'pairs': {}, 'energy': {}}
with safe_open(RDM, 'pt') as fr, safe_open(KLEIN, 'pt') as fk:
    for k in fr.keys():
        full = fr.get_tensor(k).float()
        for dk, sl in pieces(k):
            w0 = fk.get_tensor(dk).float()
            if sl == 'swap':
                h = full.shape[0] // 2
                w1 = torch.cat([full[h:], full[:h]])
            elif sl is not None:
                w1 = full[sl[0]:sl[1]]
            else:
                w1 = full
            assert w1.shape == w0.shape, (k, dk, w1.shape, w0.shape)
            d = w1 - w0
            cos = torch.nn.functional.cosine_similarity(w1.reshape(1, -1), w0.reshape(1, -1)).item()
            rel = (d.norm() / w0.norm()).item()
            report['pairs'][dk] = dict(cos=round(cos, 5), rel_change=round(rel, 5))
            if cos < 0.9:
                print(f'!! {k} -> {dk}: cos {cos:.3f}', flush=True)
            if d.ndim == 2 and any(dk.endswith(t + ('' if t.endswith('.weight') else '.weight')) for t in TARGET):
                U, S, Vh = torch.linalg.svd(d, full_matrices=False)
                e = (S ** 2).cumsum(0) / (S ** 2).sum()
                report['energy'][dk] = {r: round(e[min(r, len(e)) - 1].item(), 4) for r in RANKS}
                sq = S[:R].sqrt()
                mod = 'transformer.' + dk.removesuffix('.weight')
                lora[f'{mod}.lora_A.weight'] = (sq[:, None] * Vh[:R]).to(torch.bfloat16).contiguous()
                lora[f'{mod}.lora_B.weight'] = (U[:, :R] * sq[None, :]).to(torch.bfloat16).contiguous()
                print(f'{dk}: change {rel:.4f}, rank energy ' +
                      ' '.join(f'{r}:{report["energy"][dk][r]:.3f}' for r in RANKS), flush=True)
            else:
                rest[dk] = w1.to(torch.bfloat16).contiguous()
                print(f'{dk}: change {rel:.4f} (kept exactly)', flush=True)

save_file(lora, f'{OUT}/rdm_lora_r{R}.safetensors', metadata={'rank': str(R), 'alpha': str(R), 'source': 'epfl-vita/flux2-klein-1step-rdm minus black-forest-labs/FLUX.2-klein-4B'})
save_file(rest, f'{OUT}/rdm_other.safetensors')
json.dump(report, open(f'{OUT}/delta_report.json', 'w'), indent=1)
E = report['energy']
for r in RANKS:
    print(f'rank {r}: mean energy {sum(v[r] for v in E.values()) / len(E):.3f}, min {min(v[r] for v in E.values()):.3f}')
