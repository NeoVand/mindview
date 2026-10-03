<script lang="ts">
	import { labGPU, labModel, labPainter, labPainting, labScheduler } from '$lib/lab/shared';
	import { Stage } from '$lib/lab/stage';
	import { Cube, flopsText, type CubeLabel } from '$lib/lab/cube';
	import type { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import type { Painter } from '$lib/runtime/painter';
	import Status from '$lib/ui/Status.svelte';
	import NeedsComputer from '$lib/ui/NeedsComputer.svelte';
	import { mayLoad } from '$lib/lab/device';
	import FastForward from '@lucide/svelte/icons/fast-forward';
	import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
	import Scan from '@lucide/svelte/icons/scan';
	import SlidersHorizontal from '@lucide/svelte/icons/sliders-horizontal';

	let cube: Cube | undefined;
	let stage: Stage | undefined;
	let loading = $state('Starting the graphics card');
	let progress = $state(0);
	let ready = $state(false);
	let error = $state<string | null>(null);
	let blocked = $state<{ why: string; go: () => void } | null>(null); // the models may not fit here
	let prompt = $state('a bonsai tree made of glowing circuitry in a dark museum, volumetric light');
	let busy = $state('');
	let labels = $state<CubeLabel[]>([]);
	let stretch = $state(32);
	let playing = $state(false);
	let follow = $state(false);
	let totals = $state({ words: 0, flops: 0, attn: 0 });
	let painterTotals = $state<{ all: number; attn: number; reader: number } | null>(null);
	let painterNote = $state('');
	let painting = $state({ at: '', done: false });
	let llm: BonsaiLLM | undefined;
	let painter: Painter | undefined;
	let tip = $state<{ x: number; y: number; title: string; lines: string[] } | null>(null);
	let more = $state(true); // on a phone the choices under the prompt can be folded away

	function mount(canvas: HTMLCanvasElement) {
		let alive = true;
		let off: (() => void) | undefined;
		const hover = (e: PointerEvent) => {
			if (!cube || e.buttons) return;
			const b = cube.boxAt(e.clientX, e.clientY);
			const where = b?.painter
				? b.painter.step < 0
					? ', before the steps'
					: `, step ${b.painter.step + 1}${b.painter.block >= 0 && b.painter.block < 25 ? `, block ${b.painter.block + 1}` : ''}`
				: b && b.layer >= 0
					? `, layer ${b.layer + 1}`
					: '';
			tip = b
				? {
						x: e.clientX,
						y: e.clientY,
						title: `${b.name}${where}`,
						lines: [
							`${flopsText(b.flops)} operations (a multiply and an add each count as one)`,
							...(b.painter && b.painter.step >= 0
								? [
										'1,536 rows deep: the prompt’s 512, then the picture’s 1,024 patches; lit: the rows the painting keeps'
									]
								: [])
						]
					}
				: null;
		};
		const leave = () => (tip = null);
		canvas.addEventListener('pointermove', hover);
		canvas.addEventListener('pointerleave', leave);
		(async () => {
			// the labs keep about 3 GB on the GPU, the reader's per-layer states in one buffer of 521 MB: on a phone, ask first
			await mayLoad(3, 521, (why, go) => (blocked = { why, go }));
			blocked = null;
			const gpu = await labGPU(canvas);
			loading = 'Loading Ternary Bonsai 1.7B (460 MB, kept after the first time)';
			llm = await labModel(gpu.device, (f) => (progress = f));
			if (!alive) return;
			stage = new Stage(gpu);
			// the painting goes on while this lab is open (it is shared with the other labs)
			stage.scheduler = labScheduler(gpu.device);
			cube = new Cube(stage, llm);
			stage.onFrame = () => {
				if (!cube) return;
				labels = cube.labels();
				if (cube.busy !== busy) busy = cube.busy;
				if (cube.playing !== playing) playing = cube.playing;
				const run = cube.pc?.run;
				if (run) {
					const at = run.done
						? ''
						: run.at.block < 0 && run.at.step === 0
							? 'reading the prompt'
							: `step ${run.at.step + 1}, block ${Math.min(25, run.at.block + 1)}`;
					if (at !== painting.at || run.done !== painting.done) painting = { at, done: run.done };
				}
			};
			stage.start(cube);
			off = stage.attachControls(canvas, (x, y) => {
				const b = cube?.boxAt(x, y);
				if (b && cube) cube.focus(cube.boxes.indexOf(b));
			});
			if (import.meta.env.DEV) (window as unknown as { cube: Cube }).cube = cube;
			ready = true;
			await read();
			// then the painter: its boxes go below the reader's and light as it paints
			painterNote = 'Loading the painter (1.1 GB, kept after the first time)';
			painter = await labPainter(gpu.device, (p) => {
				painterNote = `Loading the painter: ${Math.round(p.fraction * 100)}%`;
			});
			if (!alive || !cube) return;
			painterNote = 'Measuring the painter’s weights';
			await cube.attachPainter(painter, labPainting(gpu.device, llm, painter, prompt.trim()));
			painterNote = '';
			painterTotals = cube.painterFlops;
		})().catch((e) => (error = e instanceof Error ? e.message : String(e)));
		return () => {
			alive = false;
			off?.();
			canvas.removeEventListener('pointermove', hover);
			canvas.removeEventListener('pointerleave', leave);
			stage?.destroy();
			cube?.destroy();
		};
	}

	async function read() {
		if (!cube) return;
		if (painter && llm && stage)
			cube.setRun(labPainting(stage.device, llm, painter, prompt.trim()));
		await cube.read(prompt.trim());
		totals = { words: cube.wordCount, flops: cube.totalFlops, attn: cube.attnFlops };
		painterTotals = cube.painterFlops;
		cube.focus(0);
		setTimeout(() => cube?.overview(), 6000);
	}

	function submit(e?: SubmitEvent) {
		e?.preventDefault();
		if (prompt.trim()) read();
	}

	function promptKey(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submit();
		}
	}
</script>

<svelte:head><title>Compute cube</title></svelte:head>

<main class="lab-main">
	<canvas
		class="lab-canvas"
		{@attach mount}
		aria-label="Every multiplication of the forward pass as a voxel; drag to turn, right-drag or two fingers to move, scroll or pinch to come closer"
	></canvas>
	<div class="lab-labels" aria-hidden="true">
		{#each labels as l, i (i)}
			<span
				style:left="{l.x}px"
				style:top="{l.y}px"
				style:font-size="{l.size}px"
				style:opacity={l.alpha}>{l.text}</span
			>
		{/each}
	</div>
	{#if blocked}
		<NeedsComputer why={blocked.why} ontry={blocked.go} />
	{:else if error}
		<Status text={error} error />
	{:else if !ready}
		<Status text={loading} {progress} />
	{:else}
		<form onsubmit={submit} class="lab-ask" class:folded={!more}>
			<div class="top">
				<textarea
					class="prompt"
					bind:value={prompt}
					onkeydown={promptKey}
					rows="1"
					aria-label="Type something for the model to read; Enter to read it"
					spellcheck="false"></textarea>
				<button
					type="button"
					class="btn quiet icon more-toggle"
					aria-label={more ? 'Hide the choices' : 'Show the choices'}
					aria-expanded={more}
					onclick={() => (more = !more)}><SlidersHorizontal /></button
				>
			</div>
			<div class="more">
				<div class="lab-options">
					<label class="field">
						Depth of each word
						<input
							type="range"
							min="1"
							max="96"
							step="1"
							bind:value={stretch}
							onchange={() => cube?.setStretch(stretch)}
						/>
						<span class="value">{stretch === 1 ? 'true proportions' : `× ${stretch}`}</span>
					</label>
					<label class="field">
						<input
							type="checkbox"
							bind:checked={follow}
							onchange={() => cube && (cube.follow = follow)}
						/>
						Follow the work
					</label>
				</div>
				<div class="lab-options">
					{#if playing}
						<button type="button" class="btn" onclick={() => cube?.skip()}
							><FastForward />Show all</button
						>
					{:else}
						<button type="button" class="btn" onclick={() => cube?.play()}
							><RotateCcw />Replay in order</button
						>
					{/if}
					<button type="button" class="btn" onclick={() => cube?.overview()}
						><Scan />Everything</button
					>
				</div>
			</div>
		</form>
		<div class="lab-bottom">
			<div class="glass lab-narrator">
				<p class="narration" aria-live="polite">
					{#if busy}
						{busy}…
					{:else if totals.words}
						Top: the reader, every multiplication for these {totals.words} tokens, one voxel each: {flopsText(
							totals.flops
						)} operations. Attention (the two boxes that grow with the square of the prompt's length)
						is
						{((totals.attn / totals.flops) * 100).toFixed(2)}% of it; the rest is weights times
						words.
						{#if painterTotals}
							Below: the painting of it, {flopsText(painterTotals.all)} operations, {Math.round(
								painterTotals.all / totals.flops
							)} times as many: the reader again at 512 tokens ({flopsText(painterTotals.reader)}),
							then the painter's 25 blocks over 1,536 rows, four times. Lit: the rows it keeps (the
							words and 17 patches){painting.done ? '' : `; it is painting now (${painting.at})`}.
						{:else if painterNote}
							{painterNote}…
						{/if}
						Click a box to fly to it.
					{/if}
				</p>
			</div>
		</div>
	{/if}
	{#if tip}
		<div class="tip glass" style:left="{tip.x + 16}px" style:top="{tip.y + 16}px">
			<strong>{tip.title}</strong>
			{#each tip.lines as line, i (i)}
				<span>{line}</span>
			{/each}
		</div>
	{/if}
</main>
