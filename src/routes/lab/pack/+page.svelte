<script lang="ts">
	// Check: the one-file model (mindview-t2i) paints what the reader and the painter's own files paint, for the same
	// prompt and seed. Compares the reader's taps, the conditioning's effect (the latents after every step) and the
	// finished pictures, and shows both.
	import { labGPU, labModel, labPainter } from '$lib/lab/shared';
	import { packedUrl } from '$lib/models';
	import { asset } from '$app/paths';
	import { page } from '$app/state';
	import { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import { PackedModel } from '$lib/runtime/packed';
	import { Painter } from '$lib/runtime/painter';

	const { NT, NI, CIN, TAP } = Painter.dims;
	let log = $state<string[]>([]);
	// ?file=<name in static/models/mindview-t2i/> picks a variant; ?prompt= and ?seed= the painting
	const q = page.url.searchParams;
	const file = q.get('file');
	const prompt =
		q.get('prompt') ?? 'a bonsai tree made of glowing circuitry in a dark museum, volumetric light';
	const seed = Number(q.get('seed') ?? 7);
	let canvases: HTMLCanvasElement[] = $state([]);
	let result = $state<Record<string, number | string> | null>(null);

	const say = (s: string) => (log = [...log, s]);

	async function pixels(device: GPUDevice, tex: GPUTexture): Promise<Uint8Array> {
		const buf = device.createBuffer({
			size: 512 * 512 * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = device.createCommandEncoder();
		enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: 2048 }, [512, 512]);
		device.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const out = new Uint8Array(buf.getMappedRange().slice(0));
		buf.destroy();
		return out;
	}

	async function paint(painter: Painter, llm: BonsaiLLM, device: GPUDevice) {
		const t0 = performance.now();
		await painter.encode(llm, prompt);
		const taps = await painter.read(painter.at.taps, NT * TAP);
		painter.setNoise(undefined, seed);
		const lats: Float32Array[] = [];
		for (let s = 0; s < 4; s++) {
			await painter.step(s);
			lats.push(await painter.read(painter.at.lat, NI * CIN));
		}
		await painter.decode();
		const px = await pixels(device, painter.taef2.texture);
		return { taps, lats, px, ms: performance.now() - t0 };
	}

	function show(i: number, px: Uint8Array) {
		const c = canvases[i];
		const ctx = c.getContext('2d')!;
		ctx.putImageData(new ImageData(new Uint8ClampedArray(px), 512, 512), 0, 0);
	}

	const maxDiff = (a: Float32Array, b: Float32Array) => {
		let m = 0,
			r = 0;
		for (let i = 0; i < a.length; i++) {
			m = Math.max(m, Math.abs(a[i] - b[i]));
			r = Math.max(r, Math.abs(a[i]));
		}
		return { abs: m, rel: m / (r || 1) };
	};

	async function run(canvas: HTMLCanvasElement) {
		const gpu = await labGPU(canvas);
		const device = gpu.device;
		say('Loading the reader and the painter from their own files');
		const llm = await labModel(device);
		const painter = await labPainter(device);
		say('Painting with them');
		const a = await paint(painter, llm, device);
		show(0, a.px);
		say(`done in ${(a.ms / 1000).toFixed(1)} s`);

		say('Loading the one-file model');
		const t0 = performance.now();
		const model = await PackedModel.open(
			file ? asset(`/models/mindview-t2i/${file}`) : packedUrl()
		);
		const llm2 = BonsaiLLM.fromGGUF(device, await model.readerGGUF());
		const t1 = performance.now();
		const painter2 = await Painter.fromPacked(device, model);
		const t2 = performance.now();
		say(
			`reader ${((t1 - t0) / 1000).toFixed(1)} s, painter ${((t2 - t1) / 1000).toFixed(1)} s (${painter2.fused ? 'fused conditioning' : 'separate'})`
		);
		say('Painting with it');
		const b = await paint(painter2, llm2, device);
		show(1, b.px);
		say(`done in ${(b.ms / 1000).toFixed(1)} s`);

		const taps = maxDiff(a.taps, b.taps);
		let se = 0;
		for (let i = 0; i < a.px.length; i++) if (i % 4 !== 3) se += (a.px[i] - b.px[i]) ** 2;
		const mse = se / ((a.px.length * 3) / 4);
		result = {
			taps_max_abs: taps.abs,
			...Object.fromEntries(
				a.lats.map((l, s) => [`latent_step${s + 1}_max_rel`, maxDiff(l, b.lats[s]).rel])
			),
			picture_psnr_db: mse === 0 ? 'identical' : 10 * Math.log10((255 * 255) / mse),
			pixels_differing: a.px.reduce((n, v, i) => n + (i % 4 !== 3 && v !== b.px[i] ? 1 : 0), 0)
		};
		say('compared');
	}

	function mount(canvas: HTMLCanvasElement) {
		run(canvas).catch((e) => say(`error: ${e instanceof Error ? e.message : e}`));
	}
</script>

<svelte:head><title>The one-file model: check</title></svelte:head>

<main>
	<canvas class="gpu" {@attach mount} width="4" height="4"></canvas>
	<h1>The one-file model against the painter's own files</h1>
	<div class="pics">
		<figure>
			<canvas bind:this={canvases[0]} width="512" height="512"></canvas>
			<figcaption>reader + painter files</figcaption>
		</figure>
		<figure>
			<canvas bind:this={canvases[1]} width="512" height="512"></canvas>
			<figcaption>mindview-t2i (one file)</figcaption>
		</figure>
	</div>
	{#if result}
		<pre id="result">{JSON.stringify(result, null, 2)}</pre>
	{/if}
	<ol>
		{#each log as l, i (i)}<li>{l}</li>{/each}
	</ol>
</main>

<style>
	main {
		padding: 2rem;
		color: var(--bone);
		background: var(--void);
		min-height: 100vh;
	}
	.gpu {
		position: absolute;
		width: 4px;
		height: 4px;
		opacity: 0;
	}
	h1 {
		font-weight: 300;
		font-size: 1.4rem;
	}
	.pics {
		display: flex;
		gap: 1.5rem;
		flex-wrap: wrap;
	}
	figure {
		margin: 0;
	}
	figure canvas {
		width: 384px;
		height: 384px;
		display: block;
	}
	pre,
	ol {
		font-size: 0.85rem;
	}
</style>
