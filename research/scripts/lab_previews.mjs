// The pictures on /lab: each lab running, its controls hidden, saved as static/previews/<lab>.png (convert to WebP).
// usage (with `pnpm dev` running): node research/scripts/lab_previews.mjs <out dir> [lab=seconds ...]
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const out = process.argv[2];
const plan = process.argv.slice(3).length
	? process.argv.slice(3).map((a) => a.split('='))
	: [
			['threads', '120'],
			['layer', '40'],
			['machine', '120'],
			['cube', '120']
		];
fs.mkdirSync(out, { recursive: true });
const ctx = await chromium.launchPersistentContext(
	process.env.PROFILE ?? path.join(out, '.profile'),
	{
		executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: true,
		ignoreDefaultArgs: ['--enable-unsafe-swiftshader'],
		args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist']
	}
);
const page = await ctx.newPage();
await page.setViewportSize({ width: 1200, height: 750 });
page.on('pageerror', (e) => console.log('page error:', e.message));
for (const [lab, secs] of plan) {
	await page.goto(`http://localhost:5173/lab/${lab}`);
	await page.waitForTimeout(Number(secs) * 1000);
	await page.addStyleTag({
		content: `header.site, .lab-ask, .lab-bottom, .journey, .ask, .stations, .status, .tip { visibility: hidden !important; }`
	});
	await page.waitForTimeout(500);
	await page.screenshot({ path: path.join(out, `${lab}.png`) });
	console.log(lab);
}
await ctx.close();
