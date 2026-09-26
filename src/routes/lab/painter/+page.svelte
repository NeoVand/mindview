<script lang="ts">
	import { asset } from '$app/paths';
	import { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import { Painter } from '$lib/runtime/painter';
	import { PainterFiles } from '$lib/runtime/painter-files';
	import { Taef2 } from '$lib/runtime/taef2';

	let log = $state<string[]>([]);
	let busy = $state(false);
	let view: HTMLCanvasElement | undefined;
	const say = (s: string) => (log = [...log, s]);

	async function device() {
		const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
		if (!adapter) throw new Error('No WebGPU adapter.');
		return adapter.requestDevice({
			requiredLimits: {
				maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
				maxBufferSize: adapter.limits.maxBufferSize
			}
		});
	}

	const cosine = (a: Float32Array, b: Float32Array) => {
		let ab = 0,
			aa = 0,
			bb = 0;
		for (let i = 0; i < a.length; i++) {
			ab += a[i] * b[i];
			aa += a[i] * a[i];
			bb += b[i] * b[i];
		}
		return ab / Math.sqrt(aa * bb);
	};
	const bin = async (name: string) =>
		new Float32Array(await (await fetch(asset(`/lab/painter_ref/${name}.bin`))).arrayBuffer());

	/** Run the whole painter on the reference prompt and noise, comparing with PyTorch at every block. */
	async function checkPainter() {
		busy = true;
		try {
			const dev = await device();
			const ref = await (await fetch(asset('/lab/painter_ref/ref.json'))).json();
			let t0 = performance.now();
			const [llm, painter] = await Promise.all([
				BonsaiLLM.load(dev, asset('/models/ternary-bonsai-1.7b/model.gguf')),
				Painter.load(dev, asset('/models/bonsai-image-4b'))
			]);
			say(`loaded the 1.7B and the painter in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
			if (import.meta.env.DEV) Object.assign(window, { painter, llm, dev });
			dev.pushErrorScope('validation');
			t0 = performance.now();
			const enc = await painter.encode(llm, ref.prompt);
			const sameIds =
				enc.ids.length === ref.ids.length &&
				enc.ids.every((v: number, i: number) => v === ref.ids[i]);
			const condRef = await bin('cond_rows');
			let worst = 1;
			for (const [i, r] of (ref.rows as number[]).entries()) {
				const ours = await painter.read(painter.at.ctx + r * 7680, 7680);
				worst = Math.min(worst, cosine(ours, condRef.subarray(i * 7680, (i + 1) * 7680)));
			}
			say(
				`text: ${enc.real} real tokens, ids ${sameIds ? 'identical' : 'DIFFERENT'}; conditioning worst row cosine ${worst.toFixed(6)} (${((performance.now() - t0) / 1000).toFixed(1)} s)`
			);
			painter.setNoise(await bin('noise'));
			const blocks = await bin('step1_blocks'); // [26][16][3072]
			const probe = ref.probe_rows as number[];
			const vel = await bin('velocity');
			const lines: string[] = [];
			for (let s = 0; s < 4; s++) {
				t0 = performance.now();
				await painter.step(s, async (_, b) => {
					if (s > 0 || b > 24) return;
					// step 1: compare the probe rows of the residual stream after every block
					let c = 1;
					for (const [i, r] of probe.entries()) {
						const ours = await painter.read(painter.at.h + r * 3072, 3072);
						c = Math.min(
							c,
							cosine(
								ours,
								blocks.subarray(((b + 1) * 16 + i) * 3072, ((b + 1) * 16 + i + 1) * 3072)
							)
						);
					}
					if (b % 4 === 0 || b === 24 || b < 0 || c < 0.999)
						say(`step 1 block ${b + 1}: worst probe-row cosine ${c.toFixed(6)}`);
				});
				const v = await painter.read(painter.at.vel, 1024 * 128);
				lines.push(
					`step ${s + 1}: ${((performance.now() - t0) / 1000).toFixed(1)} s, velocity cosine ${cosine(v, vel.subarray(s * 131072, (s + 1) * 131072)).toFixed(6)}`
				);
				say(lines[lines.length - 1]);
			}
			const err = await dev.popErrorScope();
			if (err) say(`WebGPU: ${err.message}`);
			t0 = performance.now();
			await painter.decode();
			say(`decoded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
			const rd = dev.createBuffer({
				size: 512 * 512 * 4,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
			});
			const e2 = dev.createCommandEncoder();
			e2.copyTextureToBuffer(
				{ texture: painter.taef2.texture },
				{ buffer: rd, bytesPerRow: 2048 },
				[512, 512]
			);
			dev.queue.submit([e2.finish()]);
			await rd.mapAsync(GPUMapMode.READ);
			const px = new Uint8ClampedArray(rd.getMappedRange().slice(0));
			view?.getContext('2d')?.putImageData(new ImageData(px, 512, 512), 0, 0);
		} catch (e) {
			say(`error: ${e instanceof Error ? e.message : String(e)}`);
		}
		busy = false;
	}

	/** Decode a reference latent with the WebGPU TAEF2 and compare with PyTorch's pixels. */
	async function checkDecoder() {
		busy = true;
		try {
			const dev = await device();
			const files = await PainterFiles.open(asset('/models/bonsai-image-4b'));
			await files.fetch(['taef2.bin']);
			const weights = new Map(files.names('taef2.').map((k) => [k.slice(6), files.dense(k)]));
			const tae = new Taef2(dev, weights, 512);
			if (import.meta.env.DEV) Object.assign(window, { tae, dev, taeW: weights });
			const latent = new Float32Array(
				await (await fetch(asset('/lab/taef2_ref/latent.bin'))).arrayBuffer()
			);
			const ref = new Float32Array(
				await (await fetch(asset('/lab/taef2_ref/out.bin'))).arrayBuffer()
			);
			tae.writeLatent(latent);
			dev.pushErrorScope('validation');
			const t0 = performance.now();
			const enc = dev.createCommandEncoder();
			tae.decode(enc);
			const rd = dev.createBuffer({
				size: 512 * 512 * 4,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
			});
			enc.copyTextureToBuffer(
				{ texture: tae.texture },
				{ buffer: rd, bytesPerRow: 2048 },
				[512, 512]
			);
			dev.queue.submit([enc.finish()]);
			await dev.queue.onSubmittedWorkDone();
			const ms = performance.now() - t0;
			const err = await dev.popErrorScope();
			if (err) throw new Error(err.message);
			await rd.mapAsync(GPUMapMode.READ);
			const px = new Uint8Array(rd.getMappedRange().slice(0));
			let worst = 0,
				sum = 0;
			for (let c = 0; c < 3; c++)
				for (let i = 0; i < 512 * 512; i++) {
					const want = Math.min(1, Math.max(0, ref[c * 512 * 512 + i]));
					const d = Math.abs(px[i * 4 + c] / 255 - want);
					worst = Math.max(worst, d);
					sum += d;
				}
			say(
				`TAEF2 decode in ${ms.toFixed(0)} ms; max pixel error ${(worst * 255).toFixed(1)}/255, mean ${((sum / (3 * 512 * 512)) * 255).toFixed(2)}/255`
			);
			const ctx = view?.getContext('2d');
			if (ctx) ctx.putImageData(new ImageData(new Uint8ClampedArray(px.buffer), 512, 512), 0, 0);
		} catch (e) {
			say(`error: ${e instanceof Error ? e.message : String(e)}`);
		}
		busy = false;
	}
</script>

<svelte:head><title>Painter runtime check</title></svelte:head>

<main>
	<h1>The painter, running in this tab</h1>
	<button onclick={checkDecoder} disabled={busy}>Check the decoder against PyTorch</button>
	<button onclick={checkPainter} disabled={busy}>Paint the reference prompt and compare</button>
	<ol>
		{#each log as line, i (i)}
			<li>{line}</li>
		{/each}
	</ol>
	<div class="pair">
		<canvas
			{@attach (c: HTMLCanvasElement) => {
				view = c;
			}}
			width="512"
			height="512"
			aria-label="Decoded in this tab"
		></canvas>
		<img
			src={asset('/lab/painter_ref/final.png')}
			width="512"
			height="512"
			alt="Painted by PyTorch"
		/>
	</div>
</main>

<style>
	main {
		padding: 4vh 6vw;
		font-size: 0.95rem;
		line-height: 1.5;
		height: 100vh;
		overflow: auto;
		box-sizing: border-box;
	}
	h1 {
		font-weight: 300;
		font-style: italic;
		font-size: 1.6rem;
	}
	button {
		font: inherit;
		background: none;
		color: var(--bone);
		border: 1px solid rgb(232 226 214 / 0.3);
		padding: 0.4rem 0.8rem;
	}
	.pair {
		display: flex;
		gap: 1rem;
		flex-wrap: wrap;
	}
	canvas,
	img {
		width: min(512px, 42vw);
		height: auto;
	}
</style>
