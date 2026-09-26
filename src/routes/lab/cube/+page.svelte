<script lang="ts">
	import { labGPU, labModel, labPainter, labPainting, labScheduler } from '$lib/lab/shared';
	import { Stage } from '$lib/lab/stage';
	import { Cube, flopsText, type CubeLabel } from '$lib/lab/cube';
	import type { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import type { Painter } from '$lib/runtime/painter';

	let cube: Cube | undefined;
	let stage: Stage | undefined;
	let loading = $state('Starting the graphics card');
	let progress = $state(0);
	let ready = $state(false);
	let error = $state<string | null>(null);
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

<main>
	<canvas
		{@attach mount}
		aria-label="Every multiplication of the forward pass as a voxel; drag to turn, right-drag to move, scroll to come closer"
	></canvas>
	<div class="labels" aria-hidden="true">
		{#each labels as l, i (i)}
			<span
				style:left="{l.x}px"
				style:top="{l.y}px"
				style:font-size="{l.size}px"
				style:opacity={l.alpha}>{l.text}</span
			>
		{/each}
	</div>
	{#if error}
		<p class="note" role="alert">{error}</p>
	{:else if !ready}
		<div class="note">
			<p>{loading}</p>
			<div class="bar"><span style:transform="scaleX({progress})"></span></div>
		</div>
	{:else}
		<form onsubmit={submit}>
			<textarea
				bind:value={prompt}
				onkeydown={promptKey}
				rows="1"
				aria-label="Type something for the model to read; Enter to read it"
				spellcheck="false"></textarea>
			<div class="options">
				<label>
					Depth of each word
					<input
						type="range"
						min="1"
						max="96"
						step="1"
						bind:value={stretch}
						onchange={() => cube?.setStretch(stretch)}
					/>
					<span class="num">{stretch === 1 ? 'true proportions' : `× ${stretch}`}</span>
				</label>
				<label>
					<input
						type="checkbox"
						bind:checked={follow}
						onchange={() => cube && (cube.follow = follow)}
					/>
					Follow the work
				</label>
				{#if playing}
					<button type="button" onclick={() => cube?.skip()}>Show all</button>
				{:else}
					<button type="button" onclick={() => cube?.play()}>Replay in order</button>
				{/if}
				<button type="button" onclick={() => cube?.overview()}>Everything</button>
			</div>
		</form>
		<p class="caption" aria-live="polite">
			{#if busy}
				{busy}…
			{:else if totals.words}
				Top: the reader, every multiplication for these {totals.words} tokens, one voxel each: {flopsText(
					totals.flops
				)} operations. Attention (the two boxes that grow with the square of the prompt's length) is
				{((totals.attn / totals.flops) * 100).toFixed(2)}% of it; the rest is weights times words.
				{#if painterTotals}
					Below: the painting of it, {flopsText(painterTotals.all)} operations, {Math.round(
						painterTotals.all / totals.flops
					)} times as many: the reader again at 512 tokens ({flopsText(painterTotals.reader)}), then
					the painter's 25 blocks over 1,536 rows, four times. Lit: the rows it keeps (the words and
					17 patches){painting.done ? '' : `; it is painting now (${painting.at})`}.
				{:else if painterNote}
					{painterNote}…
				{/if}
				Click a box to fly to it.
			{/if}
		</p>
	{/if}
	{#if tip}
		<div class="tip" style:left="{tip.x + 16}px" style:top="{tip.y + 16}px">
			<strong>{tip.title}</strong>
			{#each tip.lines as line, i (i)}
				<span>{line}</span>
			{/each}
		</div>
	{/if}
</main>

<style>
	main {
		position: fixed;
		inset: 0;
		background: var(--void);
	}
	canvas {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		touch-action: none;
		cursor: grab;
	}
	canvas:active {
		cursor: grabbing;
	}
	.labels span {
		position: absolute;
		transform: translate(-50%, -50%);
		color: var(--bone);
		white-space: nowrap;
		pointer-events: none;
		font-weight: 300;
	}
	form {
		position: absolute;
		left: 2.2rem;
		top: 3.2rem;
		width: min(46rem, 70vw);
	}
	textarea {
		display: block;
		width: 100%;
		max-height: 20vh;
		field-sizing: content;
		resize: none;
		background: none;
		border: none;
		border-bottom: 1px solid rgb(232 226 214 / 0.2);
		color: var(--bone);
		font: italic 300 clamp(1.05rem, 1.5vw, 1.6rem) / 1.3 var(--serif);
		padding: 0.3rem 0;
		outline: none;
	}
	textarea:focus-visible {
		border-bottom-color: var(--ember);
	}
	.options {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.8rem 1.4rem;
		margin-top: 0.8rem;
		font-size: 0.82rem;
	}
	.options label {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		opacity: 0.8;
	}
	.num {
		font-variant-numeric: tabular-nums;
		min-width: 7rem;
	}
	input[type='range'] {
		accent-color: var(--ember);
		width: 8rem;
	}
	input[type='checkbox'] {
		accent-color: var(--ember);
	}
	button {
		background: none;
		border: 1px solid rgb(232 226 214 / 0.3);
		color: var(--bone);
		font: inherit;
		font-size: 0.82rem;
		padding: 0.25rem 0.7rem;
		cursor: pointer;
	}
	button:focus-visible,
	input:focus-visible {
		outline: 1px solid var(--ember);
		outline-offset: 2px;
	}
	.caption,
	.note {
		position: absolute;
		left: 2.2rem;
		bottom: 2rem;
		max-width: 64ch;
		margin: 0;
		font-weight: 300;
		line-height: 1.5;
		color: var(--bone);
		opacity: 0.8;
		pointer-events: none;
	}
	.bar {
		height: 1px;
		width: min(28rem, 60vw);
		margin-top: 0.8rem;
		background: rgb(232 226 214 / 0.15);
	}
	.bar span {
		display: block;
		height: 100%;
		background: var(--ember);
		transform-origin: left;
	}
	.tip {
		position: absolute;
		z-index: 10;
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 0.55rem 0.75rem;
		background: rgb(10 10 10 / 0.88);
		border: 1px solid rgb(232 226 214 / 0.18);
		color: var(--bone);
		font-size: 0.82rem;
		pointer-events: none;
	}
	.tip strong {
		font-weight: 400;
		color: var(--ember);
	}
</style>
