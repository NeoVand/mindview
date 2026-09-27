"""EPFL's 1-step fine-tune on the ternary painter: what does it need, and at what rank?

rdm_test.py showed that the ternary DiT plus RDM's change from klein paints in 1 step, and that rank 256 for the 100
big matrices is as good as the whole change. RDM's change has two kinds of part:
    matrices   the 100 big linears (0.17% change each), as a low-rank side branch
    time       what depends only on the timestep: the time embedder, the three modulation linears, the final layer's
               modulation (0.1-0.6%); the runtime precomputes all of these per step, so they cost nothing
    embed      x_embedder, context_embedder, proj_out (0.08-0.3%)
Columns (1 step from sigma 1, 512 x 512, stock-encoder prompts, one seed): time only, time + embed, and time + embed
with the matrices at rank 32, 64 and 128. Each change is added to the ternary weights (RDM - klein).

    python scripts/rdm_rank.py [n_prompts]
"""
import glob
import os
import sys

import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from PIL import Image
from safetensors import safe_open

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED = 7
OUT = 'renders/rdm_rank'
os.makedirs(OUT, exist_ok=True)
HUB = '/Users/neo/.cache/huggingface/hub/'
TERN = glob.glob(HUB + 'models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*/')[0]
KLEIN = glob.glob(HUB + 'models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/')[0]
LOWRANK = 'data/rdm/rdm_lora_r256.safetensors'
OTHER = 'data/rdm/rdm_other.safetensors'  # RDM's own values of everything but the 100 matrices
TIME = ('time_guidance_embed.', 'double_stream_modulation_img.', 'double_stream_modulation_txt.',
        'single_stream_modulation.', 'norm_out.linear.')
EMBED = ('x_embedder.', 'context_embedder.', 'proj_out.')

cache = [d for d in (torch.load(f, map_location='cpu') for f in sorted(glob.glob('data/prompt_cache/*.pt')))
         if d['encoder'] == 'qwen'][:n_prompts]
pipe = Flux2KleinPipeline.from_pretrained(TERN, text_encoder=None, transformer=None, torch_dtype=torch.bfloat16)
dev, dt = 'mps', torch.bfloat16
pipe.vae.to(dev)

# RDM - klein for the small parts
small = {}
with safe_open(OTHER, 'pt') as fo, safe_open(KLEIN + 'transformer/diffusion_pytorch_model.safetensors', 'pt') as fk:
    for k in fo.keys():
        if k.startswith(TIME + EMBED):
            small[k] = fo.get_tensor(k).float() - fk.get_tensor(k).float()
print('small parts', sorted({k.split('.')[0] for k in small}), flush=True)


def load(parts, rank):
    tr = Flux2Transformer2DModel.from_pretrained(TERN, subfolder='transformer', torch_dtype=dt)
    params = dict(tr.named_parameters())
    for k, d in small.items():
        if k.startswith(parts):
            params[k].data = (params[k].data.float() + d).to(dt)
    if rank:
        with safe_open(LOWRANK, 'pt') as s:
            for m in {k.rsplit('.lora_', 1)[0] for k in s.keys()}:
                p = params[m.removeprefix('transformer.') + '.weight']
                B = s.get_tensor(f'{m}.lora_B.weight').float()[:, :rank]
                A = s.get_tensor(f'{m}.lora_A.weight').float()[:rank]
                p.data = (p.data.float() + B @ A).to(dt)
    return tr.to(dev)


def decode(x0):
    lat = x0.permute(0, 2, 1).reshape(1, 128, 32, 32)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(dev, torch.float32)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev, torch.float32)
    lat = pipe._unpatchify_latents(lat * std + mean)
    with torch.no_grad():
        img = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type='pil')[0]


CONFIGS = [('time', TIME, 0), ('time+embed', TIME + EMBED, 0), ('r32', TIME + EMBED, 32), ('r64', TIME + EMBED, 64),
           ('r128', TIME + EMBED, 128)]
if os.environ.get('CONFIGS'):  # e.g. CONFIGS=r8,r16,r32-time (-time: without the embedders' change)
    CONFIGS = [(c, TIME if c.endswith('-time') else TIME + EMBED, int(c[1:].removesuffix('-time')))
               for c in os.environ['CONFIGS'].split(',')]
for label, parts, rank in CONFIGS:
    tr = load(parts, rank)
    for i, d in enumerate(cache):
        emb = d['embeds'].to(dev, dt)
        z = torch.randn((1, 128, 32, 32), generator=torch.Generator('cpu').manual_seed(SEED)).to(dev)
        eps, ids = pipe._pack_latents(z), pipe._prepare_latent_ids(z).to(dev)
        with torch.no_grad():
            v = tr(hidden_states=eps.to(dt), timestep=torch.tensor([1.0], device=dev, dtype=dt), guidance=None,
                   encoder_hidden_states=emb, txt_ids=pipe._prepare_text_ids(emb).to(dev), img_ids=ids,
                   return_dict=False)[0].float()
        decode(eps - v).save(f'{OUT}/p{i}_{label}.png')
        print(label, i, flush=True)
    del tr
    torch.mps.empty_cache()
# the sheet: these, then rdm_test's rank 256 and today's 2-step Fast (SANA-Sprint LoRA)
cols = [(c, OUT) for c, _, _ in CONFIGS] + [('tern+r256', 'renders/rdm_test'), ('sana-2', 'renders/rdm_test')]
suffix = os.environ.get('CONFIGS', '').replace(',', '_')
sheet = Image.new('RGB', (len(cols) * 256, len(cache) * 256))
for i in range(len(cache)):
    for j, (c, where) in enumerate(cols):
        sheet.paste(Image.open(f'{where}/p{i}_{c}.png').resize((256, 256)), (j * 256, i * 256))
sheet.save(f'{OUT}/sheet{suffix and "_" + suffix}.jpg', quality=88)
print('columns:', [c for c, _ in cols])
