"""Schedules for other step counts: the painter's modulation depends only on the timestep, so for each step count
the sigmas (FLUX.2 klein's empirical shift for 512 x 512) and the modulation vectors are precomputed here with
diffusers' own modules, exactly as export_painter.py does for 4 steps.

Writes static/models/bonsai-image-4b/schedules.bin (f32) + schedules.json:
  { "512": { "<steps>": { sigmas: [...], offset: <floats>, rows: 17 } } }  each step: [17, 3072] =
  double_img (6) | double_txt (6) | single (3) | norm_out (2), in the same order the browser keeps them.
"""
import glob, json, math, os
import numpy as np
import torch
from safetensors import safe_open
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
from diffusers.models.transformers.transformer_flux2 import Flux2TimestepGuidanceEmbeddings, Flux2Modulation
from diffusers.models.normalization import AdaLayerNormContinuous

OUT = '../static/models/bonsai-image-4b'
ROOT = glob.glob('/Users/neo/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
cfg = json.load(open(ROOT + 'transformer/config.json'))
DT = safe_open(ROOT + 'transformer/diffusion_pytorch_model.safetensors', 'pt')
D = 3072


def load_module(mod, prefix):
    mod.load_state_dict({k: DT.get_tensor(f'{prefix}.{k}').float() for k in mod.state_dict()})
    return mod.float().eval()


temb_mod = load_module(Flux2TimestepGuidanceEmbeddings(cfg['timestep_guidance_channels'], D, bias=False, guidance_embeds=False), 'time_guidance_embed')
m_img = load_module(Flux2Modulation(D, 2), 'double_stream_modulation_img')
m_txt = load_module(Flux2Modulation(D, 2), 'double_stream_modulation_txt')
m_single = load_module(Flux2Modulation(D, 1), 'single_stream_modulation')
norm_out = load_module(AdaLayerNormContinuous(D, D, elementwise_affine=False, eps=cfg['eps'], bias=False), 'norm_out')
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(ROOT + 'scheduler')

meta, blobs, offset = {}, [], 0
for size in [512]:
    n_img = (size // 16) ** 2
    meta[str(size)] = {}
    for steps in [1, 2, 3, 4, 6, 8, 12]:
        mu = compute_empirical_mu(image_seq_len=n_img, num_steps=steps)
        sched.set_timesteps(steps, 'cpu', sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
        sigmas = sched.sigmas.float()
        t_model = ((sched.timesteps.float() / 1000).float() * 1000).float()
        with torch.no_grad():
            temb = temb_mod(t_model, None)
            rows = torch.cat([m_img(temb).reshape(-1, 6, D), m_txt(temb).reshape(-1, 6, D), m_single(temb).reshape(-1, 3, D),
                              norm_out.linear(norm_out.silu(temb)).reshape(-1, 2, D)], 1)   # [steps, 17, D]
        a = rows.numpy().astype('<f4')
        blobs.append(a.tobytes())
        meta[str(size)][str(steps)] = dict(sigmas=sigmas.tolist(), mu=mu, offset=offset, rows=17)
        offset += a.size
        print(size, steps, 'sigmas', [round(x, 4) for x in sigmas.tolist()])
# '1r': 1 step with EPFL's 1-step fine-tune (epfl-vita/flux2-klein-1step-rdm): the same modules plus RDM's change from
# klein (RDM - klein, added to the ternary model's weights; see rdm_delta.py), at sigma 1
RDM_OTHER, KLEIN = 'data/rdm/rdm_other.safetensors', glob.glob(
    '/Users/neo/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/transformer/')
if os.path.exists(RDM_OTHER) and KLEIN:
    fo, fk = safe_open(RDM_OTHER, 'pt'), safe_open(KLEIN[0] + 'diffusion_pytorch_model.safetensors', 'pt')

    def load_rdm(mod, prefix):
        mod.load_state_dict({k: DT.get_tensor(f'{prefix}.{k}').float() + fo.get_tensor(f'{prefix}.{k}').float()
                             - fk.get_tensor(f'{prefix}.{k}').float() for k in mod.state_dict()})
        return mod.float().eval()

    r_temb = load_rdm(Flux2TimestepGuidanceEmbeddings(cfg['timestep_guidance_channels'], D, bias=False, guidance_embeds=False), 'time_guidance_embed')
    r_img, r_txt = load_rdm(Flux2Modulation(D, 2), 'double_stream_modulation_img'), load_rdm(Flux2Modulation(D, 2), 'double_stream_modulation_txt')
    r_single = load_rdm(Flux2Modulation(D, 1), 'single_stream_modulation')
    r_out = load_rdm(AdaLayerNormContinuous(D, D, elementwise_affine=False, eps=cfg['eps'], bias=False), 'norm_out')
    with torch.no_grad():
        temb = r_temb(torch.tensor([1000.0]), None)
        rows = torch.cat([r_img(temb).reshape(-1, 6, D), r_txt(temb).reshape(-1, 6, D), r_single(temb).reshape(-1, 3, D),
                          r_out.linear(r_out.silu(temb)).reshape(-1, 2, D)], 1)
    a = rows.numpy().astype('<f4')
    blobs.append(a.tobytes())
    meta['512']['1r'] = dict(sigmas=[1.0, 0.0], mu=None, offset=offset, rows=17,
                             source='epfl-vita/flux2-klein-1step-rdm (its change from klein, on the ternary weights)')
    offset += a.size
    print('1r: RDM 1-step modulation')
open(f'{OUT}/schedules.bin', 'wb').write(b''.join(blobs))
json.dump(meta, open(f'{OUT}/schedules.json', 'w'), indent=1)
print('wrote schedules', offset * 4 / 1e6, 'MB')

# check: the 4-step vectors must equal the ones exported with the model
M = json.load(open(f'{OUT}/manifest.json'))
t = M['tensors']['mod.double_img']
raw = np.memmap(f'{OUT}/{t["file"]}', dtype=np.uint8, mode='r')
ref = np.frombuffer(raw, dtype=np.float32, count=int(np.prod(t['shape'])), offset=t['offset']).reshape(t['shape'])
ours = np.frombuffer(b''.join(blobs), dtype='<f4')[meta['512']['4']['offset']:].reshape(-1)[: 4 * 17 * D].reshape(4, 17, D)[:, :6]
print('4-step double_img vs the model export: max abs diff', float(np.abs(ours - ref).max()))
