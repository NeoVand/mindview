<script lang="ts">
	// Where a painting step's GPU time goes: every op of one step timed on its own (timestamp queries), summed by kind.
	// ?runs= how many profiled steps (the first warms up and is dropped); ?file= a model in static/models/mindview-t2i/;
	// ?steps= the schedule (1 or 2 turn the few-step LoRA on when the file has it); ?rows= text rows (a number or auto).
	import { asset } from '$app/paths';
	import { page } from '$app/state';
	import { labDevice } from '$lib/lab/shared';
	import { packedUrl } from '$lib/models';
	import { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import { PackedModel } from '$lib/runtime/packed';
	import { Painter } from '$lib/runtime/painter';

	const q = page.url.searchParams;
	const runs = Number(q.get('runs') ?? 3);
	const file = q.get('file');
	const steps = Number(q.get('steps') ?? 4);
	const textRows = q.get('rows') ?? '512';
	let status = $state('Loading the model');
	let report = $state('');

	async function run() {
		const device = await labDevice();
		const packed = await PackedModel.open(
			file ? asset(`/models/mindview-t2i/${file}`) : packedUrl()
		);
		const llm = BonsaiLLM.fromGGUF(device, await packed.readerGGUF());
		const painter = await Painter.fromPacked(device, packed);
		painter.setSteps(steps);
		painter.textLength = textRows === 'auto' ? 'auto' : Number(textRows);
		status = 'Reading the prompt';
		await painter.encode(llm, 'a lighthouse on a cliff during a thunderstorm, oil painting');
		painter.setNoise(undefined, 7);
		const all: { label: string; ms: number }[][] = [];
		for (let r = 0; r < runs; r++) {
			status = `Profiling step ${r + 1} of ${runs}`;
			all.push(await painter.profile(0));
		}
		const kept = all.slice(1);
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- a local tally, not state
		const by = new Map<string, { ms: number; n: number }>();
		for (const ops of kept)
			for (const o of ops) {
				const e = by.get(o.label) ?? { ms: 0, n: 0 };
				e.ms += o.ms / kept.length;
				e.n += 1 / kept.length;
				by.set(o.label, e);
			}
		const total = [...by.values()].reduce((a, e) => a + e.ms, 0);
		const rows = [...by.entries()].sort((a, b) => b[1].ms - a[1].ms);
		const kind = (l: string) =>
			l.startsWith('gemm')
				? 'ternary gemm'
				: l.startsWith('dgemm') && l.includes(' x24')
					? 'attention matmuls'
					: /^dgemm \d+x64x/.test(l)
						? 'lora A x'
						: l.startsWith('dgemm')
							? 'dense gemm'
							: l;
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- a local tally, not state
		const kinds = new Map<string, number>();
		for (const [l, e] of rows) kinds.set(kind(l), (kinds.get(kind(l)) ?? 0) + e.ms);
		const pct = (ms: number) => `${((100 * ms) / total).toFixed(1)}%`.padStart(6);
		report = [
			`one step, mean of ${kept.length}: ${total.toFixed(0)} ms of GPU time over ${Math.round(
				[...by.values()].reduce((a, e) => a + e.n, 0)
			)} ops`,
			'',
			'by kind:',
			...[...kinds.entries()]
				.sort((a, b) => b[1] - a[1])
				.map(([k, ms]) => `  ${k.padEnd(22)} ${ms.toFixed(0).padStart(6)} ms ${pct(ms)}`),
			'',
			'by op (count, total, each):',
			...rows.map(
				([l, e]) =>
					`  ${l.padEnd(30)} ${String(Math.round(e.n)).padStart(4)} ${e.ms.toFixed(0).padStart(6)} ms ${pct(e.ms)}  ${(e.ms / e.n).toFixed(2).padStart(7)} ms`
			)
		].join('\n');
		status = 'done';
	}

	function mount() {
		run().catch((e) => (status = `error: ${e instanceof Error ? e.message : e}`));
	}
</script>

<svelte:head><title>Profile a step</title></svelte:head>

<main {@attach mount}>
	<p>{status}</p>
	{#if report}<pre id="result">{report}</pre>{/if}
</main>

<style>
	main {
		position: fixed;
		inset: 0;
		overflow: auto;
		padding: 1.5rem;
		background: var(--void);
		color: var(--bone);
	}
	pre {
		font-size: 0.8rem;
		line-height: 1.4;
	}
</style>
