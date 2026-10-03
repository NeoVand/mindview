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
	import Status from '$lib/ui/Status.svelte';
	import NeedsComputer from '$lib/ui/NeedsComputer.svelte';
	import { mayLoad } from '$lib/lab/device';
	import ChevronLeft from '@lucide/svelte/icons/chevron-left';
	import ChevronRight from '@lucide/svelte/icons/chevron-right';
	import Pause from '@lucide/svelte/icons/pause';
	import Play from '@lucide/svelte/icons/play';
	import BookOpenText from '@lucide/svelte/icons/book-open-text';
	import Paintbrush from '@lucide/svelte/icons/paintbrush';
	import SlidersHorizontal from '@lucide/svelte/icons/sliders-horizontal';

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
	let blocked = $state<{ why: string; go: () => void } | null>(null); // the models may not fit here
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
	let more = $state(true); // on a phone the choices under the prompt can be folded away

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
			// the labs keep about 3 GB on the GPU: on a phone, ask first
			await mayLoad(3, 890, (why, go) => (blocked = { why, go }));
			blocked = null;
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

<main class="lab-main">
	<canvas
		class="lab-canvas"
		{@attach mount}
		aria-label="One word or patch through one layer of the reader or one block of the painter, every number; drag to turn, right-drag or two fingers to move, scroll or pinch to come closer"
	></canvas>
	<div class="lab-labels" aria-hidden="true">
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
					aria-label="Type something for the models to read and paint; Enter to start"
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
				<div class="seg" role="group" aria-label="Which model">
					<button type="button" aria-pressed={mode === 'reader'} onclick={toReader}
						><BookOpenText /><span class="long">The reader:</span> one layer</button
					>
					<button type="button" aria-pressed={mode === 'painter'} onclick={toPainter}
						><Paintbrush /><span class="long">The painter:</span> one block</button
					>
				</div>
				<div class="chips" role="group" aria-label="Choose the word to follow">
					{#each words as w, i (i)}
						<button
							type="button"
							class="chip"
							aria-pressed={mode === 'painter' ? isWord && row === 3 + i : i === chosen}
							onclick={() => choose(i)}>{w}</button
						>
					{/each}
				</div>
				{#if mode === 'reader'}
					<div class="lab-options">
						<label class="field">
							Layer
							<input type="range" min="1" max="28" bind:value={layer} onchange={setLayer} />
							<span class="value">{layer}</span>
						</label>
					</div>
				{:else}
					<div class="lab-options">
						<div class="patches" role="group" aria-label="Or a patch of the picture">
							<span class="field">or a patch</span>
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
						<label class="field">
							Block
							<input type="range" min="1" max="25" bind:value={blockNo} onchange={setBlock} />
							<span class="value">{blockNo}{blockNo <= 5 ? ', double' : ', single'}</span>
						</label>
						<div class="field steps">
							<span>Step</span>
							<div class="seg" role="group" aria-label="The step kept in full">
								{#each Array.from({ length: steps }, (_, i) => i + 1) as s (s)}
									<button type="button" aria-pressed={s === paintStep} onclick={() => setStep(s)}
										>{s}</button
									>
								{/each}
							</div>
						</div>
					</div>
				{/if}
			</div>
		</form>
		<ol class="stations" aria-label="The steps of the layer">
			{#each stations as s, i (i)}
				<li>
					<button
						type="button"
						aria-current={i === station ? 'step' : undefined}
						onclick={() => go(i)}><span class="n">{i + 1}</span>{s}</button
					>
				</li>
			{/each}
		</ol>
		<div class="lab-bottom">
			<div class="glass lab-narrator">
				{#if stations.length}<p class="where">
						<span class="n">{station + 1} of {stations.length}</span>{stations[station]}
					</p>{/if}
				<p class="narration" aria-live="polite">
					{painterNote && mode === 'painter' ? painterNote : busy ? `${busy}…` : caption}
				</p>
			</div>
			<div class="step">
				<button
					type="button"
					class="btn icon"
					onclick={() => go(station - 1)}
					aria-label="Previous step"
					title="Previous step (←)"><ChevronLeft /></button
				>
				<button
					type="button"
					class="btn icon"
					aria-label={playing ? 'Pause' : 'Play'}
					title={playing ? 'Pause (space)' : 'Play (space)'}
					onclick={() => {
						const s = scene();
						if (s) s.playing = !s.playing;
					}}
				>
					{#if playing}<Pause />{:else}<Play />{/if}
				</button>
				<button
					type="button"
					class="btn icon"
					onclick={() => go(station + 1)}
					aria-label="Next step"
					title="Next step (→)"><ChevronRight /></button
				>
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

<style>
	.lab-ask {
		width: min(40rem, calc(100% - 2 * var(--gutter) - 14rem));
	}
	.patches {
		display: flex;
		align-items: center;
		gap: 0.6rem;
	}
	.grid {
		position: relative;
		width: 3.4rem;
		height: 3.4rem;
		border: 1px solid var(--hair-2);
		border-radius: 4px;
	}
	.grid button {
		position: absolute;
		width: 0.62rem;
		height: 0.62rem;
		padding: 0;
		border-radius: 50%;
		border: 1px solid rgb(236 229 216 / 0.55);
		background: none;
		cursor: pointer;
		transform: translate(-50%, -50%);
	}
	.grid button.on {
		background: var(--ember);
		border-color: var(--ember);
	}
	.steps .seg button {
		padding: 0 0.7rem;
	}
	.stations {
		position: absolute;
		right: calc(var(--gutter) - 0.5rem);
		top: var(--top);
		margin: 0;
		padding: 0;
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 1px;
	}
	.stations button {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		width: 100%;
		padding: 0.3rem 0.6rem;
		border: 0;
		border-radius: var(--r-sm);
		background: none;
		color: var(--bone-3);
		font: 400 var(--t-sm) / 1.2 var(--sans);
		text-align: left;
		cursor: pointer;
	}
	.stations button:hover {
		color: var(--bone);
		background: var(--wash);
	}
	.stations .n,
	.where .n {
		min-width: 1.2rem;
		color: var(--bone-3);
		font-variant-numeric: tabular-nums;
		text-align: right;
	}
	.stations button[aria-current='step'] {
		color: var(--bone);
		background: var(--ember-wash);
	}
	.stations button[aria-current='step'] .n {
		color: var(--ember);
	}
	.where {
		display: none;
		gap: 0.5rem;
		margin: 0 0 0.35rem;
		color: var(--bone-2);
		font: 500 var(--t-xs) / 1.3 var(--sans);
	}
	.where .n {
		min-width: 0;
		text-align: left;
	}
	.step {
		display: flex;
		gap: 0.4rem;
		flex-shrink: 0;
	}
	@media (max-width: 900px) {
		.lab-ask {
			width: calc(100% - 2 * var(--gutter));
		}
		.stations {
			display: none;
		}
		.where {
			display: flex;
		}
	}
	@media (max-width: 720px) {
		.step {
			justify-content: center;
		}
		.seg .long {
			display: none;
		}
	}
</style>
