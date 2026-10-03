// Screenshots of every page at a desktop and a phone size, for design reviews.
// usage (with `pnpm dev` running): node research/scripts/survey_pages.mjs <out dir> [base url] [routes...]
// CLICK=<button name> presses that button after loading, and SHOTS=5,20 takes more shots that many seconds after it.
// SIZES=desk,phone picks the sizes.
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const out = process.argv[2];
const base = process.argv[3] ?? 'http://localhost:5173';
const routes = process.argv.slice(4).length
	? process.argv.slice(4)
	: ['/', '/paint', '/lab', '/lab/threads', '/lab/layer', '/lab/machine', '/lab/cube'];
fs.mkdirSync(out, { recursive: true });
const sizes = {
	desk: { viewport: { width: 1440, height: 900 } },
	phone: {
		viewport: { width: 390, height: 844 },
		deviceScaleFactor: 2,
		isMobile: true,
		hasTouch: true
	}
};
const ctx = await chromium.launchPersistentContext(
	process.env.PROFILE ?? path.join(out, '.profile'),
	{
		executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: true,
		ignoreDefaultArgs: ['--enable-unsafe-swiftshader'],
		args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist']
	}
);
const only = (process.env.SIZES ?? 'desk,phone').split(',');
for (const [name, opts] of Object.entries(sizes).filter(([n]) => only.includes(n))) {
	const page = await ctx.newPage();
	await page.setViewportSize(opts.viewport);
	if (name === 'phone') {
		// a phone's user agent (the pages ask a phone before loading the models)
		const cdp = await ctx.newCDPSession(page);
		await cdp.send('Emulation.setUserAgentOverride', {
			userAgent:
				'Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 Mobile/15E148 Safari/604.1'
		});
		await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
	}
	page.on('pageerror', (e) => console.log(name, 'page error:', e.message));
	page.on(
		'console',
		(m) => m.type() === 'error' && console.log(name, 'console:', m.text().slice(0, 200))
	);
	for (const r of routes) {
		await page.goto(base + r);
		await page.waitForTimeout(Number(process.env.WAIT ?? 3000));
		const file = path.join(out, `${name}${r.replaceAll('/', '_') || '_'}.png`);
		await page.screenshot({ path: file });
		console.log(file);
		if (process.env.CLICK) {
			await page.getByRole('button', { name: process.env.CLICK }).first().click();
			const t0 = Date.now();
			for (const at of (process.env.SHOTS ?? '').split(',').filter(Boolean).map(Number)) {
				await page.waitForTimeout(Math.max(0, at * 1000 - (Date.now() - t0)));
				const f = file.replace(/\.png$/, `-${at}s.png`);
				await page.screenshot({ path: f });
				console.log(f);
			}
		}
	}
	await page.close();
}
await ctx.close();
