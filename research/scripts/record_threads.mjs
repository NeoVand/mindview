// Record the landing's first run: the threads piece reading and painting its opening prompt live, on the one-file model
// in static/models/mindview-t2i/model.gguf, in headless Chrome on this machine's GPU; its files (recording.json,
// recording.bin, the pictures as WebP) go to static/recordings/bonsai-museum, which the landing plays back.
// usage (with `pnpm dev` running): node research/scripts/record_threads.mjs [base url] [out dir]
// The painter's events keep the times they happened at, so record on a cool machine with nothing else on the GPU.
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const base = process.argv[2] ?? 'http://localhost:5173';
const out =
	process.argv[3] ?? path.join(import.meta.dirname, '../../static/recordings/bonsai-museum');
const browser = await chromium.launch({
	executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
	headless: true,
	ignoreDefaultArgs: ['--enable-unsafe-swiftshader'],
	args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist']
});
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
page.on('pageerror', (e) => console.log('page error:', e.message));
page.on(
	'console',
	(m) => m.type() === 'error' && console.log('console error:', m.text().slice(0, 300))
);
await page.goto(`${base}/?record`);
await page.getByRole('button', { name: 'Read and paint' }).click();
const t0 = Date.now();
let files = null;
while (!files) {
	if (Date.now() - t0 > 15 * 60e3) throw new Error('no recording after 15 minutes');
	await page.waitForTimeout(2000);
	files = await page.evaluate(() => window.recordingFiles?.());
}
fs.mkdirSync(out, { recursive: true });
let bytes = 0;
for (const [name, b64] of Object.entries(files)) {
	const buf = Buffer.from(b64, 'base64');
	fs.writeFileSync(path.join(out, name), buf);
	bytes += buf.length;
	console.log(name, (buf.length / 1e3).toFixed(0), 'kB');
}
console.log(
	`recorded in ${((Date.now() - t0) / 1000).toFixed(0)} s: ${(bytes / 1e6).toFixed(2)} MB in ${out}`
);
await browser.close();
