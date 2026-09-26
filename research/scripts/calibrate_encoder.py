"""Fit a per-channel affine map from Ternary-Bonsai-4B hidden states to Qwen3-4B's (layers 9/18/27),
so the ternary LLM can stand in as Bonsai Image's text encoder. Fit on calibration prompts, apply to others."""
import hashlib, torch
key = lambda e, p: 'data/prompt_cache/' + hashlib.sha1(f'{e}|{p}'.encode()).hexdigest()[:12] + '.pt'
cal = [l.strip() for l in open('data/calibration_prompts.txt') if l.strip()]
X = torch.cat([torch.load(key('bonsai4b', p))['embeds'][0].float() for p in cal])  # [N, 7680]
Y = torch.cat([torch.load(key('qwen', p))['embeds'][0].float() for p in cal])
mx, my = X.mean(0), Y.mean(0)
xc, yc = X - mx, Y - my
a = (xc * yc).sum(0) / ((xc * xc).sum(0) + 1e-3)
b = my - a * mx
pred = a * X + b
r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - my) ** 2).sum()
print('per-channel affine R^2 on calibration set:', round(r2.item(), 3))
torch.save(dict(a=a, b=b), 'data/bonsai4b_to_qwen_affine.pt')
test = ["a bonsai tree made of glowing circuitry in a dark museum, volumetric light", "a portrait of an old fisherman, weathered skin, soft window light",
        "an ocean wave frozen in glass at sunrise", "a red fox sleeping in fresh snow"]
for p in test:
    d = torch.load(key('bonsai4b', p)); y = torch.load(key('qwen', p))['embeds'].float()
    e = (a * d['embeds'].float() + b)
    cos = torch.nn.functional.cosine_similarity(e[0, 3:d['n_real']], y[0, 3:d['n_real']], dim=-1).mean()
    print(f'held-out cos after calibration {cos:.3f}  ({p[:30]})')
    d['embeds'] = e.to(torch.bfloat16); d['encoder'] = 'bonsai4b_cal'
    torch.save(d, key('bonsai4b_cal', p))
