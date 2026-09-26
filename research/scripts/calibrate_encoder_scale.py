"""Magnitude-only calibration: per tap layer (9/18/27), scale Bonsai-4B hidden states so their mean real-token
norm matches Qwen3-4B's on the calibration prompts. Keeps directions (meaning) intact."""
import hashlib, torch
key = lambda e, p: 'data/prompt_cache/' + hashlib.sha1(f'{e}|{p}'.encode()).hexdigest()[:12] + '.pt'
cal = [l.strip() for l in open('data/calibration_prompts.txt') if l.strip()]
ratios = torch.zeros(3)
for p in cal:
    d = torch.load(key('bonsai4b', p)); q = torch.load(key('qwen', p))
    n = d['n_real']
    for i in range(3):
        x = d['embeds'][0, :n, i * 2560:(i + 1) * 2560].float(); y = q['embeds'][0, :n, i * 2560:(i + 1) * 2560].float()
        ratios[i] += (y.norm(dim=-1).mean() / x.norm(dim=-1).mean()) / len(cal)
print('scale per tap layer (qwen/bonsai4b):', ratios.tolist())
test = ["a bonsai tree made of glowing circuitry in a dark museum, volumetric light", "a portrait of an old fisherman, weathered skin, soft window light",
        "an ocean wave frozen in glass at sunrise", "a red fox sleeping in fresh snow"]
for p in test:
    d = torch.load(key('bonsai4b', p))
    e = d['embeds'].float().reshape(1, 512, 3, 2560) * ratios.view(1, 1, 3, 1)
    d['embeds'] = e.reshape(1, 512, 7680).to(torch.bfloat16); d['encoder'] = 'bonsai4b_scale'
    torch.save(d, key('bonsai4b_scale', p))
