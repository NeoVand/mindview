<script lang="ts">
	// Side by side: the same prompts and seed painted by the reader and painter files, and by one-file model variants
	// (?files=a.gguf,b.gguf from static/models/mindview-t2i/). Each model is loaded, paints every prompt, and is freed
	// before the next, so only one is on the GPU at a time.
	import { asset } from '$app/paths';
	import { page } from '$app/state';
	import { labGPU } from '$lib/lab/shared';
	import { painterUrl, readerUrl } from '$lib/models';
	import { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import { PackedModel } from '$lib/runtime/packed';
	import { Painter } from '$lib/runtime/painter';

	const PROMPTS = [
		'a bonsai tree made of glowing circuitry in a dark museum, volumetric light',
		'a red fox sleeping in the snow at dawn',
		'three apples and one pear on a wooden table',
		'a cat wearing a tiny astronaut helmet, studio photo',
		'a lighthouse on a cliff during a thunderstorm, oil painting',
		'a bowl of ramen with a soft boiled egg, top view',
		'a neon sign that says OPEN in a rainy window',
		'a blue cube on top of a red sphere'
	];
	const q = page.url.searchParams;
	const files = (q.get('files') ?? '').split(',').filter(Boolean);
	const seed = Number(q.get('seed') ?? 7);
	const columns = ['reader + painter files', ...files];
	let cells = $state<Record<string, string>>({}); // `${column}|${prompt}` -> data URL
	let status = $state('Starting');
	let done = $state(false);

	async function pixels(device: GPUDevice, tex: GPUTexture): Promise<string> {
		const buf = device.createBuffer({
			size: 512 * 512 * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = device.createCommandEncoder();
		enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: 2048 }, [512, 512]);
		device.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const px = new Uint8ClampedArray(buf.getMappedRange().slice(0));
		buf.destroy();
		const c = new OffscreenCanvas(512, 512);
		c.getContext('2d')!.putImageData(new ImageData(px, 512, 512), 0, 0);
		const blob = await c.convertToBlob({ type: 'image/jpeg', quality: 0.9 });
		return URL.createObjectURL(blob);
	}

	async function paintAll(col: string, llm: BonsaiLLM, painter: Painter, device: GPUDevice) {
		for (const p of PROMPTS) {
			status = `${col}: ${p}`;
			await painter.encode(llm, p);
			painter.setNoise(undefined, seed);
			for (let s = 0; s < 4; s++) await painter.step(s);
			await painter.decode();
			cells[`${col}|${p}`] = await pixels(device, painter.taef2.texture);
		}
	}

	async function run(canvas: HTMLCanvasElement) {
		const { device } = await labGPU(canvas);
		status = 'Loading the reader and painter files';
		{
			const llm = await BonsaiLLM.load(device, readerUrl());
			const painter = await Painter.load(device, painterUrl());
			await paintAll(columns[0], llm, painter, device);
			llm.destroy();
			painter.destroy();
		}
		for (const f of files) {
			status = `Loading ${f}`;
			const model = await PackedModel.open(asset(`/models/mindview-t2i/${f}`));
			const llm = BonsaiLLM.fromGGUF(device, await model.readerGGUF());
			const painter = await Painter.fromPacked(device, model);
			await paintAll(f, llm, painter, device);
			llm.destroy();
			painter.destroy();
		}
		status = 'done';
		done = true;
	}

	function mount(canvas: HTMLCanvasElement) {
		run(canvas).catch((e) => (status = `error: ${e instanceof Error ? e.message : e}`));
	}
</script>

<svelte:head><title>The one-file model: grid</title></svelte:head>

<main>
	<canvas class="gpu" {@attach mount} width="4" height="4"></canvas>
	<p class="status">{status}</p>
	<table>
		<thead>
			<tr>
				<th></th>
				{#each columns as c (c)}<th>{c}</th>{/each}
			</tr>
		</thead>
		<tbody>
			{#each PROMPTS as p (p)}
				<tr>
					<td class="prompt">{p}</td>
					{#each columns as c (c)}
						<td>
							{#if cells[`${c}|${p}`]}<img src={cells[`${c}|${p}`]} alt="{c}: {p}" />{/if}
						</td>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
	{#if done}<p id="done">done</p>{/if}
</main>

<style>
	main {
		padding: 1rem;
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
	.status {
		font-size: 0.85rem;
		opacity: 0.7;
	}
	table {
		border-collapse: collapse;
	}
	th {
		font-weight: 400;
		font-size: 0.8rem;
		padding: 0.3rem;
	}
	td {
		padding: 3px;
		vertical-align: middle;
	}
	td.prompt {
		width: 11rem;
		font-size: 0.8rem;
		font-style: italic;
	}
	img {
		width: 200px;
		height: 200px;
		display: block;
	}
</style>
