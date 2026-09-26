// Headless look at /lab/threads: screenshots at chosen captions / times.
// usage: node shoot.mjs <outdir> <spec...>   spec = "wait:<regex>" | "sleep:<s>" | "shot:<name>" | "click:<x>,<y>" |
//        "scroll:<x>,<y>,<dy>" | "drag:<x0>,<y0>,<x1>,<y1>"
import { chromium } from '/Users/neo/repos/mindview/node_modules/playwright/index.mjs';

const [out, ...specs] = process.argv.slice(2);
const ctx = await chromium.launchPersistentContext(out + '/../profile', {
	executablePath:
		'/Users/neo/Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
	headless: true,
	viewport: { width: 1280, height: 800 },
	args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal', '--ignore-gpu-blocklist']
});
const page = ctx.pages()[0] ?? (await ctx.newPage());
page.on('console', (m) => {
	if (m.type() === 'error' || m.type() === 'warning') console.log('console', m.type(), m.text().slice(0, 300));
});
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.goto('http://localhost:5173/lab/threads');
const caption = () => page.evaluate(() => document.querySelector('main')?.innerText ?? '');
const t0 = Date.now();
for (const s of specs) {
	const [k, v] = [s.slice(0, s.indexOf(':')), s.slice(s.indexOf(':') + 1)];
	if (k === 'wait') {
		const re = new RegExp(v);
		for (;;) {
			const c = await caption();
			if (re.test(c)) break;
			if (Date.now() - t0 > 400_000) {
				console.log('timeout waiting for', v, '\n', c.slice(-300));
				break;
			}
			await page.waitForTimeout(250);
		}
	} else if (k === 'sleep') await page.waitForTimeout(Number(v) * 1000);
	else if (k === 'shot') {
		await page.screenshot({ path: `${out}/${v}.png` });
		const c = await caption();
		console.log(`${((Date.now() - t0) / 1000).toFixed(1)}s ${v}: ${c.split('\n').slice(-3).join(' | ').slice(0, 200)}`);
	} else if (k === 'click') {
		const [x, y] = v.split(',').map(Number);
		await page.mouse.click(x, y);
	} else if (k === 'scroll') {
		const [x, y, dy] = v.split(',').map(Number);
		await page.mouse.move(x, y);
		await page.mouse.wheel(0, dy);
	} else if (k === 'drag') {
		const [x0, y0, x1, y1] = v.split(',').map(Number);
		await page.mouse.move(x0, y0);
		await page.mouse.down();
		await page.mouse.move(x1, y1, { steps: 10 });
		await page.mouse.up();
	}
}
await ctx.close();
