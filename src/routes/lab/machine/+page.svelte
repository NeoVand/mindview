<script lang="ts">
	import { labGPU, labModel, labPainter, labPainting, labScheduler } from '$lib/lab/shared';
	import type { Painter } from '$lib/runtime/painter';
	import { Stage } from '$lib/lab/stage';
	import { Machine, type CellInfo, type MachineLabel } from '$lib/lab/machine';
	import Status from '$lib/ui/Status.svelte';
	import NeedsComputer from '$lib/ui/NeedsComputer.svelte';
	import { mayLoad } from '$lib/lab/device';
	import Activity from '@lucide/svelte/icons/activity';
	import Grid3x3 from '@lucide/svelte/icons/grid-3x3';
	import Sun from '@lucide/svelte/icons/sun';
	import FastForward from '@lucide/svelte/icons/fast-forward';
	import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
	import Scan from '@lucide/svelte/icons/scan';
	import SlidersHorizontal from '@lucide/svelte/icons/sliders-horizontal';

	let machine: Machine | undefined;
	let stage: Stage | undefined;
	let loading = $state('Starting the graphics card');
	let progress = $state(0);
	let ready = $state(false);
	let error = $state<string | null>(null);
	let blocked = $state<{ why: string; go: () => void } | null>(null); // the models may not fit here
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
	let more = $state(true); // on a phone the choices under the prompt can be folded away
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
			// the labs keep about 3 GB on the GPU: on a phone, ask first
			await mayLoad(3, 890, (why, go) => (blocked = { why, go }));
			blocked = null;
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
				if (machine.step + 1 !== step) step = machine.step + 1;
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

<main class="lab-main">
	<canvas
		class="lab-canvas"
		{@attach mount}
		aria-label="Every weight of the model on one wall; drag to turn, right-drag or two fingers to move, scroll or pinch to come closer"
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
				<div class="chips" role="group" aria-label="Choose the word whose computation is shown">
					{#each words as w, i (i)}
						<button type="button" class="chip" aria-pressed={i === chosen} onclick={() => choose(i)}
							>{w}</button
						>
					{/each}
				</div>
				<div class="lab-options">
					<div class="seg" role="group" aria-label="What the cells show">
						<button
							type="button"
							aria-pressed={mode === 'work'}
							onclick={() => {
								mode = 'work';
								machine?.setMode('work');
							}}><Activity />Weights at work</button
						>
						<button
							type="button"
							aria-pressed={mode === 'weights'}
							onclick={() => {
								mode = 'weights';
								machine?.setMode('weights');
							}}><Grid3x3 />Weights alone</button
						>
					</div>
					<label class="field">
						<input
							type="checkbox"
							bind:checked={sorted}
							onchange={() => machine?.setSorted(sorted)}
						/>
						Gather the busiest lines
					</label>
					<label class="field" title="Brightness">
						<Sun size={16} />
						<span class="sr-only">Brightness</span>
						<input
							type="range"
							min="0.03"
							max="3"
							step="0.01"
							bind:value={exposure}
							oninput={() => machine?.setExposure(exposure)}
						/>
					</label>
				</div>
				<div class="lab-options">
					{#if revealing}
						<button type="button" class="btn" onclick={() => machine?.skip()}
							><FastForward />Show all</button
						>
					{:else}
						<button type="button" class="btn" onclick={() => machine?.play()}
							><RotateCcw />Replay in order</button
						>
					{/if}
					<button type="button" class="btn" onclick={() => machine?.overview()}
						><Scan />Whole machine</button
					>
					<div class="field">
						<span>Painting step</span>
						<div class="seg" role="group" aria-label="The painting step shown">
							{#each [1, 2, 3, 4] as n (n)}
								<button type="button" aria-pressed={step === n} onclick={() => setStep(n)}
									>{n}</button
								>
							{/each}
						</div>
					</div>
				</div>
				<p class="hint">
					{following === 'word'
						? 'Following the word through the painter. Click the finished picture to follow a patch of it.'
						: 'Following a patch of the picture (●). Click another, or a word above, to follow it instead.'}
				</p>
			</div>
		</form>
		<div class="lab-bottom">
			<div class="glass lab-narrator">
				<p class="narration" aria-live="polite">
					{#if busy}
						{busy}…
					{:else if mode === 'work'}
						Left, the reader's 1.4 billion ternary weights; right, the painter's 3.7 billion, lit as
						it paints. Each cell is a weight times the number it multiplies for
						{following === 'word' ? `“${words[chosen]}”` : 'the chosen patch of the picture'}: a
						word goes through the reader, the adapter and the painter's word weights; a patch
						through the picture weights; both through the single blocks. Scroll into any cell to
						read it.
						{#if prediction && following === 'word'}The reader's own guess for the next word:
							<em>{prediction}</em>.{/if}
					{:else}
						The weights alone: −1 (blue), +1 (amber), 0 (dark), each times its group's scale. On
						their own they look like static; switch back to see them at work.
					{/if}
				</p>
				{#if painterNote || (paint && paint.text)}
					<p class="hint">{painterNote || paint?.text}</p>
					{#if paint && paint.fraction < 1}<div class="progress">
							<span style:transform="scaleX({paint.fraction})"></span>
						</div>{/if}
				{/if}
			</div>
		</div>
	{/if}
	{#if tip}
		<div class="tip glass" style:left="{tip.x + 16}px" style:top="{tip.y + 16}px">
			<strong>{tip.info.title}</strong>
			{#each tip.info.lines as line, i (i)}
				<span>{line}</span>
			{/each}
		</div>
	{/if}
</main>

<style>
	.lab-ask {
		width: min(46rem, calc(100% - 2 * var(--gutter) - 19rem));
	}
	.lab-ask .hint {
		margin: 0;
	}
	.lab-narrator .hint {
		margin: 0.5rem 0 0;
	}
	.lab-narrator .progress {
		margin-top: 0.5rem;
	}
	@media (max-width: 900px) {
		.lab-ask {
			width: calc(100% - 2 * var(--gutter));
		}
	}
</style>
