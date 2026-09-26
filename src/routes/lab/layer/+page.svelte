<script lang="ts">
	import {
		FOLLOWED_PATCHES,
		labGPU,
		labModel,
		labPainter,
		labPainting,
		labScheduler
	} from '$lib/lab/shared';
	import { Stage } from '$lib/lab/stage';
	import { Layer, type LayerLabel } from '$lib/lab/layer';
	import { Block } from '$lib/lab/block';
	import type { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import type { Painter } from '$lib/runtime/painter';

	const NT = 512;
	let lab: Layer | undefined;
	let block: Block | undefined;
	let stage: Stage | undefined;
	let llm: BonsaiLLM | undefined;
	let painter: Painter | undefined;
	let loading = $state('Starting the graphics card');
	let progress = $state(0);
	let ready = $state(false);
	let error = $state<string | null>(null);
	let prompt = $state('a bonsai tree made of glowing circuitry in a dark museum, volumetric light');
	let mode = $state<'reader' | 'painter'>('reader');
	let words = $state<string[]>([]);
	let chosen = $state(0);
	let layer = $state(5);
	let blockNo = $state(8);
	let row = $state(NT + 16 * 32 + 16);
	let paintStep = $state(4);
	let steps = $state(4);
	let painterNote = $state('');
	let stations = $state<string[]>([]);
	let station = $state(0);
	let playing = $state(false);
	let caption = $state('');
	let busy = $state('');
	let labels = $state<LayerLabel[]>([]);
	let tip = $state<{ x: number; y: number; title: string; lines: string[] } | null>(null);

	const scene = () => (mode === 'painter' && block ? block : lab);

	function mount(canvas: HTMLCanvasElement) {
		let alive = true;
		let off: (() => void) | undefined;
		const hover = (e: PointerEvent) => {
			const s = scene();
			if (!s || e.buttons) return;
			const info = s.inspect(e.clientX, e.clientY);
			tip = info ? { x: e.clientX, y: e.clientY, ...info } : null;
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
			// a painting started in another lab goes on while this one is open
			stage.scheduler = labScheduler(gpu.device);
			lab = new Layer(stage, llm);
			lab.layer = layer - 1;
			stations = lab.stationNames;
			stage.onFrame = () => {
				const s = scene();
				if (!s) return;
				labels = s.labels();
				if (s.caption !== caption) caption = s.caption;
				if (s.busy !== busy) busy = s.busy;
				if (s.station !== station) station = s.station;
				if (s.playing !== playing) playing = s.playing;
				const names = s.stationNames;
				if (names.length !== stations.length || names[0] !== stations[0]) stations = names;
				if (block && mode === 'painter' && block.row !== row) row = block.row;
			};
			stage.start(lab);
			off = stage.attachControls(canvas, (x, y) => {
				if (mode === 'painter') block?.click(x, y);
				else lab?.click(x, y);
			});
			if (import.meta.env.DEV)
				Object.assign(window as unknown as Record<string, unknown>, {
					layerLab: lab,
					layerPage: {
						toPainter,
						get block() {
							return block;
						}
					}
				});
			ready = true;
			await lab.read(prompt);
			words = lab.tokenTexts;
			chosen = lab.wordIndex;
			stage.jumpTo({ target: [0, 0, 0], dist: 30, yaw: 0, pitch: 0 });
			lab.goTo(0);
			lab.playing = true;
		})().catch((e) => (error = e instanceof Error ? e.message : String(e)));
		return () => {
			alive = false;
			off?.();
			canvas.removeEventListener('pointermove', hover);
			canvas.removeEventListener('pointerleave', leave);
			stage?.destroy();
			lab?.destroy();
			block?.destroy();
		};
	}

	/** The shared painting of this prompt, keeping everything at the chosen step. */
	function painting() {
		return labPainting(stage!.device, llm!, painter!, prompt.trim(), { fullStep: paintStep - 1 });
	}

	async function toPainter() {
		if (!stage || !llm || !lab) return;
		mode = 'painter';
		lab.playing = false;
		if (!block) {
			painterNote = 'Loading the painter (1.1 GB, kept after the first time)';
			try {
				painter = await labPainter(stage.device, (p) => {
					painterNote = `Loading the painter: ${Math.round(p.fraction * 100)}%`;
				});
			} catch (e) {
				painterNote = `The painter could not load: ${e instanceof Error ? e.message : e}`;
				return;
			}
			painterNote = '';
			block = new Block(stage, llm, painter);
			block.block = blockNo - 1;
			steps = painting().steps;
			paintStep = Math.min(paintStep, steps);
			block.row = row;
		}
		const run = painting();
		if (block.run !== run) block.setRun(run);
		if (mode !== 'painter') return;
		stage.setScene(block);
		stations = block.stationNames;
		block.goTo(0);
	}

	function toReader() {
		if (!stage || !lab) return;
		mode = 'reader';
		if (block) block.playing = false;
		stage.setScene(lab);
		stations = lab.stationNames;
		lab.goTo(lab.station);
	}

	async function submit(e?: SubmitEvent) {
		e?.preventDefault();
		if (!lab || !prompt.trim()) return;
		const reading = lab.read(prompt.trim());
		if (mode === 'painter' && block && painter) block.setRun(painting());
		await reading;
		words = lab.tokenTexts;
		chosen = lab.wordIndex;
	}

	function promptKey(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submit();
		}
	}

	function choose(i: number) {
		if (mode === 'painter') {
			block?.choose({ row: 3 + i });
			row = 3 + i;
			return;
		}
		chosen = i;
		lab?.choose(3 + i, layer - 1);
	}

	function choosePatch(p: number) {
		row = NT + p;
		block?.choose({ row: NT + p });
	}

	function setLayer() {
		lab?.choose(3 + chosen, layer - 1);
	}

	function setBlock() {
		block?.choose({ block: blockNo - 1 });
	}

	function setStep(s: number) {
		if (!block || s === paintStep) return;
		paintStep = s;
		block.setRun(painting());
	}

	function go(i: number) {
		const s = scene();
		if (!s) return;
		s.playing = false;
		s.goTo(i);
	}

	function onkeydown(e: KeyboardEvent) {
		const tag = (e.target as HTMLElement)?.tagName;
		const s = scene();
		if (!s || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
		if (e.key === 'ArrowRight') go(station + 1);
		else if (e.key === 'ArrowLeft') go(station - 1);
		else if (e.key === ' ') {
			s.playing = !s.playing;
			e.preventDefault();
		}
	}

	const isWord = $derived(row < NT);
</script>

<svelte:head><title>One layer</title></svelte:head>
<svelte:window {onkeydown} />

<main>
	<canvas
		{@attach mount}
		aria-label="One word or patch through one layer of the reader or one block of the painter, every number; drag to turn, right-drag to move, scroll to come closer"
	></canvas>
	<div class="labels" aria-hidden="true">
		{#each labels as l, i (i)}
			<span
				class={l.align ?? 'center'}
				class:italic={l.italic}
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
				aria-label="Type something for the models to read and paint; Enter to start"
				spellcheck="false"></textarea>
			<div class="modes" role="group" aria-label="Which model">
				<button type="button" class:on={mode === 'reader'} onclick={toReader}
					>The reader: one layer</button
				>
				<button type="button" class:on={mode === 'painter'} onclick={toPainter}
					>The painter: one block</button
				>
			</div>
			<div class="words" role="group" aria-label="Choose the word to follow">
				{#each words as w, i (i)}
					<button
						type="button"
						class:on={mode === 'painter' ? isWord && row === 3 + i : i === chosen}
						onclick={() => choose(i)}>{w}</button
					>
				{/each}
			</div>
			{#if mode === 'reader'}
				<div class="options">
					<label>
						Layer
						<input type="range" min="1" max="28" bind:value={layer} onchange={setLayer} />
						<span class="num">{layer}</span>
					</label>
				</div>
			{:else}
				<div class="options">
					<div class="patches" role="group" aria-label="Or a patch of the picture">
						<span class="hint">or a patch</span>
						<div class="grid">
							{#each FOLLOWED_PATCHES as p (p)}
								<button
									type="button"
									class:on={row === NT + p}
									style:left="{((p % 32) + 0.5) * (100 / 32)}%"
									style:top="{(Math.floor(p / 32) + 0.5) * (100 / 32)}%"
									aria-label="The patch at row {Math.floor(p / 32) + 1}, column {(p % 32) + 1}"
									onclick={() => choosePatch(p)}
								></button>
							{/each}
						</div>
					</div>
					<label>
						Block
						<input type="range" min="1" max="25" bind:value={blockNo} onchange={setBlock} />
						<span class="num">{blockNo}{blockNo <= 5 ? ', double' : ', single'}</span>
					</label>
					<div class="steps" role="group" aria-label="The step kept in full">
						<span>Step</span>
						{#each Array.from({ length: steps }, (_, i) => i + 1) as s (s)}
							<button type="button" class:on={s === paintStep} onclick={() => setStep(s)}
								>{s}</button
							>
						{/each}
					</div>
				</div>
			{/if}
		</form>
		<ol class="stations" aria-label="The steps of the layer">
			{#each stations as s, i (i)}
				<li>
					<button type="button" class:on={i === station} onclick={() => go(i)}>{s}</button>
				</li>
			{/each}
		</ol>
		<div class="bottom">
			<p class="caption" aria-live="polite">
				{painterNote && mode === 'painter' ? painterNote : busy ? `${busy}…` : caption}
			</p>
			<div class="step">
				<button type="button" onclick={() => go(station - 1)} aria-label="Previous step">←</button>
				<button
					type="button"
					onclick={() => {
						const s = scene();
						if (s) s.playing = !s.playing;
					}}>{playing ? 'Pause' : 'Play'}</button
				>
				<button type="button" onclick={() => go(station + 1)} aria-label="Next step">→</button>
			</div>
		</div>
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
		color: var(--bone);
		white-space: nowrap;
		pointer-events: none;
		font-weight: 300;
		transform: translate(-50%, -50%);
	}
	.labels span.left {
		transform: translate(0, -50%);
	}
	.labels span.right {
		transform: translate(-100%, -50%);
	}
	.labels span.italic {
		font-style: italic;
	}
	form {
		position: absolute;
		left: 2.2rem;
		top: 3.2rem;
		width: min(40rem, 60vw);
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
	.modes {
		display: flex;
		margin-top: 0.7rem;
	}
	.modes button + button {
		border-left: none;
	}
	.modes button.on {
		border-color: var(--ember);
		color: var(--bone);
		opacity: 1;
	}
	.modes button {
		opacity: 0.6;
	}
	.words {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem 0.5rem;
		margin-top: 0.6rem;
	}
	.words button {
		background: none;
		border: none;
		border-bottom: 1px solid transparent;
		color: var(--bone);
		opacity: 0.5;
		font: inherit;
		font-size: 0.9rem;
		padding: 0.05rem 0.1rem;
		cursor: pointer;
	}
	.words button.on {
		opacity: 1;
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
		min-width: 1.5rem;
	}
	input[type='range'] {
		accent-color: var(--ember);
		width: 10rem;
	}
	.patches {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.hint {
		opacity: 0.6;
		font-style: italic;
	}
	.grid {
		position: relative;
		width: 3.4rem;
		height: 3.4rem;
		border: 1px solid rgb(232 226 214 / 0.25);
	}
	.grid button {
		position: absolute;
		width: 0.62rem;
		height: 0.62rem;
		padding: 0;
		border-radius: 50%;
		border: 1px solid rgb(232 226 214 / 0.55);
		background: none;
		transform: translate(-50%, -50%);
	}
	.grid button.on {
		background: var(--ember);
		border-color: var(--ember);
	}
	.steps {
		display: flex;
		align-items: center;
		gap: 0.3rem;
	}
	.steps span {
		opacity: 0.8;
		margin-right: 0.2rem;
	}
	.steps button {
		padding: 0.15rem 0.5rem;
		opacity: 0.6;
	}
	.steps button.on {
		opacity: 1;
		border-color: var(--ember);
	}
	.stations {
		position: absolute;
		right: 1.4rem;
		top: 4.2rem;
		margin: 0;
		padding: 0;
		list-style: none;
		counter-reset: st;
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		font-size: 0.82rem;
	}
	.stations li {
		counter-increment: st;
		text-align: right;
	}
	.stations button {
		background: none;
		border: none;
		color: var(--bone);
		opacity: 0.4;
		font: inherit;
		padding: 0.1rem 0;
		cursor: pointer;
	}
	.stations button::after {
		content: ' ' counter(st);
		font-variant-numeric: tabular-nums;
		display: inline-block;
		width: 1.6rem;
		opacity: 0.6;
	}
	.stations button.on {
		opacity: 1;
		color: var(--ember);
	}
	.bottom {
		position: absolute;
		left: 2.2rem;
		right: 2.2rem;
		bottom: 1.8rem;
		display: flex;
		align-items: flex-end;
		justify-content: space-between;
		gap: 2rem;
	}
	.caption,
	.note {
		max-width: 68ch;
		margin: 0;
		font-weight: 300;
		line-height: 1.5;
		color: var(--bone);
		opacity: 0.82;
	}
	.note {
		position: absolute;
		left: 2.2rem;
		bottom: 2rem;
	}
	.step {
		display: flex;
		gap: 0.4rem;
		flex-shrink: 0;
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
