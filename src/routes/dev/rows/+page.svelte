<script lang="ts">
	// The one-file model painting the same prompts several ways, side by side, with the time of each painting. Each
	// column is steps/text rows: ?cols=4/512,4/auto,2/auto (1 or 2 steps turn the few-step LoRA on when the file has
	// it). ?file= picks a file in static/models/mindview-t2i/ instead of the default model.
	import { asset } from '$app/paths';
	import { page } from '$app/state';
	import { labDevice } from '$lib/lab/shared';
	import { packedUrl } from '$lib/models';
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
	const lengths = (q.get('cols') ?? '4/512,4/auto').split(',');
	const parse = (c: string) => {
		const [steps, rows] = c.split('/');
		return { steps: Number(steps), rows: rows === 'auto' ? ('auto' as const) : Number(rows) };
	};
	const file = q.get('file');
	const seed = Number(q.get('seed') ?? 7);
	let cells = $state<Record<string, { url: string; secs: number }>>({});
	let status = $state('Loading the model');
	let summary = $state('');

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
		return URL.createObjectURL(await c.convertToBlob({ type: 'image/jpeg', quality: 0.9 }));
	}

	async function run() {
		const device = await labDevice();
		const packed = await PackedModel.open(
			file ? asset(`/models/mindview-t2i/${file}`) : packedUrl()
		);
		const llm = BonsaiLLM.fromGGUF(device, await packed.readerGGUF());
		const painter = await Painter.fromPacked(device, packed);
		const totals: Record<string, number[]> = {};
		for (const p of PROMPTS)
			for (const l of lengths) {
				status = `${l}: ${p}`;
				const { steps, rows } = parse(l);
				painter.textLength = rows;
				painter.setSteps(steps);
				const t0 = performance.now();
				await painter.encode(llm, p);
				painter.setNoise(undefined, seed);
				for (let s = 0; s < steps; s++) await painter.step(s);
				await painter.decode();
				await device.queue.onSubmittedWorkDone();
				const secs = (performance.now() - t0) / 1000;
				cells[`${l}|${p}`] = { url: await pixels(device, painter.taef2.texture), secs };
				(totals[String(l)] ??= []).push(secs);
			}
		summary = Object.entries(totals)
			.map(
				([l, v]) => `${l}: ${(v.reduce((a, b) => a + b, 0) / v.length).toFixed(1)} s per picture`
			)
			.join(' · ');
		status = 'done';
	}

	function mount() {
		run().catch((e) => (status = `error: ${e instanceof Error ? e.message : e}`));
	}
</script>

<svelte:head><title>Text rows</title></svelte:head>

<main {@attach mount}>
	<p class="status">{status}</p>
	{#if summary}<p id="summary">{summary}</p>{/if}
	<table>
		<thead>
			<tr>
				<th></th>
				{#each lengths as l (l)}<th>{parse(l).steps} steps, {parse(l).rows} text rows</th>{/each}
			</tr>
		</thead>
		<tbody>
			{#each PROMPTS as p (p)}
				<tr>
					<td class="prompt">{p}</td>
					{#each lengths as l (l)}
						{@const c = cells[`${l}|${p}`]}
						<td>
							{#if c}<img src={c.url} alt="{l}: {p}" /><span>{c.secs.toFixed(1)} s</span>{/if}
						</td>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
</main>

<style>
	main {
		position: fixed;
		inset: 0;
		overflow: auto;
		padding: 1rem;
		background: var(--void);
		color: var(--bone);
	}
	.status,
	#summary {
		font-size: 0.85rem;
	}
	table {
		border-collapse: collapse;
	}
	th {
		font-weight: 400;
		font-size: 0.8rem;
	}
	td {
		padding: 3px;
		vertical-align: middle;
		font-size: 0.75rem;
	}
	td.prompt {
		width: 11rem;
		font-style: italic;
	}
	img {
		width: 200px;
		height: 200px;
		display: block;
	}
</style>
