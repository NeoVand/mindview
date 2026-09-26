<script lang="ts">
	import { labGPU, labModel, labPainter, labPainting, labScheduler } from '$lib/lab/shared';
	import type { Painter } from '$lib/runtime/painter';
	import { Stage } from '$lib/lab/stage';
	import { Machine, type CellInfo, type MachineLabel } from '$lib/lab/machine';

	let machine: Machine | undefined;
	let stage: Stage | undefined;
	let loading = $state('Starting the graphics card');
	let progress = $state(0);
	let ready = $state(false);
	let error = $state<string | null>(null);
	let prompt = $state('a bonsai tree made of glowing circuitry in a dark museum, volumetric light');
	let words = $state<string[]>([]);
	let chosen = $state(0);
	let mode = $state<'work' | 'weights'>('work');
	let sorted = $state(false);
	let exposure = $state(0.15);
	let busy = $state('');
	let prediction = $state('');
	let labels = $state<MachineLabel[]>([]);
	let tip = $state<{ x: number; y: number; info: CellInfo } | null>(null);
	let revealing = $state(false);
	let painter: Painter | undefined;
	let paint = $state<{ fraction: number; text: string } | null>(null);
	let painterNote = $state('');
	let step = $state(4);
	let following = $state<'word' | 'patch'>('word');

	function mount(canvas: HTMLCanvasElement) {
		let alive = true;
		let off: (() => void) | undefined;
		let hoverAt = 0;
		const hover = (e: PointerEvent) => {
			if (!machine || e.buttons) return;
			const now = performance.now();
			if (now - hoverAt < 60) return;
			hoverAt = now;
			const x = e.clientX,
				y = e.clientY;
			machine.inspect(x, y).then((info) => {
				tip = info ? { x, y, info } : null;
			});
		};
		const leave = () => (tip = null);
		canvas.addEventListener('pointermove', hover);
		canvas.addEventListener('pointerleave', leave);
		(async () => {
			const gpu = await labGPU(canvas);
			loading = 'Loading Ternary Bonsai 1.7B (460 MB, kept after the first time)';
			const llm = await labModel(gpu.device, (f) => (progress = f));
			if (!alive) return;
			stage = new Stage(gpu);
			machine = new Machine(stage, llm);
			machine.overview(true);
			stage.onFrame = () => {
				if (!machine) return;
				labels = machine.labels();
				if (machine.busy !== busy) busy = machine.busy;
				if (machine.prediction !== prediction) prediction = machine.prediction;
				const r = machine.playing;
				if (r !== revealing) revealing = r;
				const ps = machine.paintStatus;
				if (ps?.text !== paint?.text || ps?.fraction !== paint?.fraction) paint = ps;
			};
			stage.start(machine);
			off = stage.attachControls(canvas, (x, y) => {
				machine?.clickAt(x, y);
				if (machine) following = machine.patch === null ? 'word' : 'patch';
			});
			stage.scheduler = labScheduler(gpu.device);
			if (import.meta.env.DEV) (window as unknown as { machine: Machine }).machine = machine;
			ready = true;
			await machine.read(prompt);
			words = machine.tokenTexts;
			chosen = machine.wordIndex;
			// the painter (1.1 GB, kept after the first time), then a real painting of the same prompt
			painterNote = 'The painter is loading';
			painter = await labPainter(gpu.device, (p) => {
				painterNote =
					p.fraction < 1 ? `The painter is loading: ${Math.round(p.fraction * 100)}%` : '';
			});
			painterNote = '';
			if (!alive || !machine) return;
			machine.attachPainter(painter, labPainting(gpu.device, llm, painter, prompt));
			machine.overview();
		})().catch((e) => (error = e instanceof Error ? e.message : String(e)));
		return () => {
			alive = false;
			off?.();
			canvas.removeEventListener('pointermove', hover);
			canvas.removeEventListener('pointerleave', leave);
			stage?.destroy();
			machine?.destroy();
		};
	}

	async function submit(e?: SubmitEvent) {
		e?.preventDefault();
		if (!machine || !prompt.trim()) return;
		await machine.read(prompt.trim());
		words = machine.tokenTexts;
		chosen = machine.wordIndex;
		if (painter && stage) {
			const llm = await labModel(stage.device);
			machine.setRun(labPainting(stage.device, llm, painter, prompt.trim()));
		}
	}

	function promptKey(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submit();
		}
	}

	function choose(i: number) {
		chosen = i;
		following = 'word';
		machine?.followPatch(null);
		machine?.choose(3 + i, false);
	}

	function setStep(n: number) {
		step = n;
		machine?.setStep(n - 1);
	}
</script>

<svelte:head><title>The machine</title></svelte:head>

<main>
	<canvas
		{@attach mount}
		aria-label="Every weight of the model on one wall; drag to turn, right-drag to move, scroll to come closer"
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
			<div class="words" role="group" aria-label="Choose the word whose computation is shown">
				{#each words as w, i (i)}
					<button type="button" class:on={i === chosen} onclick={() => choose(i)}>{w}</button>
				{/each}
			</div>
			<div class="options">
				<div class="toggle" role="group" aria-label="What the cells show">
					<button
						type="button"
						class:on={mode === 'work'}
						onclick={() => {
							mode = 'work';
							machine?.setMode('work');
						}}>Weights at work</button
					>
					<button
						type="button"
						class:on={mode === 'weights'}
						onclick={() => {
							mode = 'weights';
							machine?.setMode('weights');
						}}>Weights alone</button
					>
				</div>
				<label>
					<input
						type="checkbox"
						bind:checked={sorted}
						onchange={() => machine?.setSorted(sorted)}
					/>
					Gather the busiest lines
				</label>
				<label>
					Brightness
					<input
						type="range"
						min="0.03"
						max="3"
						step="0.01"
						bind:value={exposure}
						oninput={() => machine?.setExposure(exposure)}
					/>
				</label>
				{#if revealing}
					<button type="button" onclick={() => machine?.skip()}>Show all</button>
				{:else}
					<button type="button" onclick={() => machine?.play()}>Replay in order</button>
				{/if}
				<button type="button" onclick={() => machine?.overview()}>Whole machine</button>
			</div>
			<div class="options">
				<div class="toggle" role="group" aria-label="The painting step shown">
					<span class="lead">Painting step</span>
					{#each [1, 2, 3, 4] as n (n)}
						<button type="button" class:on={step === n} onclick={() => setStep(n)}>{n}</button>
					{/each}
				</div>
				<span class="hint"
					>{following === 'word'
						? 'Following the word through the painter. Click the finished picture to follow a patch of it.'
						: 'Following a patch of the picture (●). Click another, or a word above, to follow it instead.'}</span
				>
			</div>
		</form>
		{#if painterNote || (paint && paint.text)}
			<p class="aside">
				{painterNote || paint?.text}
				{#if paint && paint.fraction < 1}<span class="bar small"
						><span style:transform="scaleX({paint.fraction})"></span></span
					>{/if}
			</p>
		{/if}
		<p class="caption" aria-live="polite">
			{#if busy}
				{busy}…
			{:else if mode === 'work'}
				Left, the reader's 1.4 billion ternary weights; right, the painter's 3.7 billion, lit as it
				paints. Each cell is a weight times the number it multiplies for
				{following === 'word' ? `“${words[chosen]}”` : 'the chosen patch of the picture'}: a word
				goes through the reader, the adapter and the painter's word weights; a patch through the
				picture weights; both through the single blocks. Scroll into any cell to read it.
				{#if prediction && following === 'word'}The reader's own guess for the next word:
					<em>{prediction}</em>.{/if}
			{:else}
				The weights alone: −1 (blue), +1 (amber), 0 (dark), each times its group's scale. On their
				own they look like static; switch back to see them at work.
			{/if}
		</p>
	{/if}
	{#if tip}
		<div class="tip" style:left="{tip.x + 16}px" style:top="{tip.y + 16}px">
			<strong>{tip.info.title}</strong>
			{#each tip.info.lines as line, i (i)}
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
		gap: 0.8rem 1.2rem;
		margin-top: 0.8rem;
		font-size: 0.82rem;
		color: var(--bone);
	}
	.options label {
		display: flex;
		align-items: center;
		gap: 0.45rem;
		opacity: 0.75;
	}
	.toggle {
		display: flex;
	}
	.toggle button {
		opacity: 0.5;
	}
	.toggle button.on {
		opacity: 1;
		border-color: var(--ember);
	}
	.toggle button + button {
		border-left: none;
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
	input[type='range'] {
		accent-color: var(--ember);
		width: 7rem;
	}
	input[type='checkbox'] {
		accent-color: var(--ember);
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
		opacity: 0.78;
		pointer-events: none;
	}
	.caption em {
		font-style: italic;
		opacity: 1;
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
		font-variant-numeric: tabular-nums;
		pointer-events: none;
		max-width: 26rem;
	}
	.aside {
		position: absolute;
		right: 1.4rem;
		top: 3.2rem;
		margin: 0;
		font-size: 0.82rem;
		color: var(--bone);
		opacity: 0.7;
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		gap: 0.4rem;
	}
	.bar.small {
		display: block;
		width: 14rem;
		margin: 0;
	}
	.lead {
		margin-right: 0.6rem;
		opacity: 0.75;
		align-self: center;
	}
	.hint {
		opacity: 0.55;
		font-style: italic;
	}
	.tip strong {
		font-weight: 400;
		color: var(--ember);
	}
</style>
