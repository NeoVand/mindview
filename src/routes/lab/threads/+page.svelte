<script lang="ts">
	import { labGPU, labModel, labPainter, releasePainting } from '$lib/lab/shared';
	import { Threads, type ThreadsStatus } from '$lib/viz/threads';

	let study: Threads | undefined;
	let loading = $state('Starting the graphics card');
	let progress = $state(0);
	let ready = $state(false);
	let error = $state<string | null>(null);
	let prompt = $state('a bonsai tree made of glowing circuitry in a dark museum, volumetric light');
	let status = $state<ThreadsStatus | null>(null);
	let painterNote = $state('');
	// the journey timeline: how far the computation has got, and where the camera is
	let front = $state(0);
	let ride = $state(0);
	let total = $state(28);
	let marks = $state<number[]>([]);
	let mode = $state('live');
	let scrubbing = false;
	// painting options (they apply to the next painting; 'Paint again' repaints with them)
	let steps = $state(4);
	let seed = $state(7);
	let stepChoices = $state([1, 2, 3, 4, 6, 8, 12]);
	let hasPainter = $state(false);

	function scrub(e: PointerEvent) {
		const box = (e.currentTarget as HTMLElement).getBoundingClientRect();
		study?.rideTo(((e.clientX - box.left) / box.width) * total);
	}

	function mount(canvas: HTMLCanvasElement) {
		let alive = true;
		let drag: { x: number; y: number; id: number } | null = null;
		const down = (e: PointerEvent) => {
			drag = { x: e.clientX, y: e.clientY, id: e.pointerId };
			canvas.setPointerCapture(e.pointerId);
		};
		const move = (e: PointerEvent) => {
			if (!drag || !study || e.pointerId !== drag.id) return;
			study.orbit(e.clientX - drag.x, e.clientY - drag.y);
			drag.x = e.clientX;
			drag.y = e.clientY;
		};
		const up = () => (drag = null);
		const wheel = (e: WheelEvent) => {
			e.preventDefault();
			study?.zoom(Math.exp(e.deltaY * 0.0015));
		};
		canvas.addEventListener('pointerdown', down);
		canvas.addEventListener('pointermove', move);
		canvas.addEventListener('pointerup', up);
		canvas.addEventListener('pointercancel', up);
		canvas.addEventListener('wheel', wheel, { passive: false });

		(async () => {
			// the same device and reader as the other labs (kept when moving between them)
			const gpu = await labGPU(canvas);
			loading = 'Loading Ternary Bonsai 1.7B (460 MB)';
			const llm = await labModel(gpu.device, (f) => (progress = f));
			if (!alive) return;
			study = new Threads(gpu, llm, (s) => {
				if (s.caption !== status?.caption || s.busy !== status?.busy) status = s;
				front = s.front ?? front;
				ride = s.ride ?? ride;
				total = s.total ?? total;
				marks = s.marks ?? marks;
				mode = s.mode ?? mode;
			});
			if (import.meta.env.DEV) (window as unknown as { threads: Threads }).threads = study;
			ready = true;
			// the painter (1.1 GB) downloads while the reader runs; the journey continues into it when it is ready
			painterNote = 'The painter is downloading';
			labPainter(gpu.device, (p) => {
				painterNote =
					p.fraction < 1 ? `The painter is downloading: ${Math.round(p.fraction * 100)}%` : '';
			})
				.then((p) => {
					painterNote = '';
					stepChoices = p.stepChoices;
					hasPainter = true;
					// the labs' shared painting uses the same painter: it stops while Threads paints
					releasePainting();
					if (alive) study?.attachPainter(p);
				})
				.catch(
					(e) => (painterNote = `The painter could not load: ${e instanceof Error ? e.message : e}`)
				);
			await study.read(prompt);
		})().catch((e) => (error = e instanceof Error ? e.message : String(e)));

		return () => {
			alive = false;
			canvas.removeEventListener('pointerdown', down);
			canvas.removeEventListener('pointermove', move);
			canvas.removeEventListener('pointerup', up);
			canvas.removeEventListener('pointercancel', up);
			canvas.removeEventListener('wheel', wheel);
			study?.destroy();
		};
	}

	function apply() {
		if (study) study.options = { steps, seed };
	}

	function submit(e?: SubmitEvent) {
		e?.preventDefault();
		apply();
		if (study && prompt.trim()) study.read(prompt.trim());
	}

	function repaint() {
		apply();
		study?.repaint();
	}

	function newSeed() {
		seed = Math.floor(Math.random() * 1e6);
		repaint();
	}

	/** Enter paints; Shift+Enter starts a new line. */
	function promptKey(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submit();
		}
	}

	function onkeydown(e: KeyboardEvent) {
		const tag = (e.target as HTMLElement)?.tagName;
		if (!study || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'BUTTON')
			return;
		if (e.key === ' ') {
			study.paused = !study.paused;
			e.preventDefault();
		} else if (e.key === 'ArrowRight') study.seek(e.shiftKey ? 10 : 2);
		else if (e.key === 'ArrowLeft') study.seek(e.shiftKey ? -10 : -2);
	}
</script>

<svelte:head><title>Threads</title></svelte:head>
<svelte:window {onkeydown} />

<main>
	<canvas
		{@attach mount}
		aria-label="Your words as threads travelling through the model's 28 layers; drag to turn, scroll to zoom"
	></canvas>
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
				aria-label="Type something for the model to read and paint; Enter to begin, Shift+Enter for a new line"
				spellcheck="false"></textarea>
			<div class="options">
				<label>
					Passes
					<select bind:value={steps} onchange={repaint} disabled={!hasPainter}>
						{#each stepChoices as n (n)}
							<option value={n}>{n}</option>
						{/each}
					</select>
				</label>
				<label>
					Seed
					<input
						type="number"
						bind:value={seed}
						min="0"
						onchange={repaint}
						disabled={!hasPainter}
					/>
				</label>
				<button type="button" onclick={newSeed} disabled={!hasPainter}>New seed</button>
				<span class="size">512 × 512 pixels</span>
			</div>
		</form>
		<p class="caption" aria-live="polite">{status?.caption ?? ''}</p>
		{#if painterNote}<p class="aside">{painterNote}</p>{/if}
		<div class="journey">
			<div
				class="track"
				role="slider"
				tabindex="0"
				aria-label="Where you are along the computation"
				aria-valuemin={0}
				aria-valuemax={total}
				aria-valuenow={Math.round(ride)}
				onpointerdown={(e) => {
					scrubbing = true;
					(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
					scrub(e);
				}}
				onpointermove={(e) => scrubbing && scrub(e)}
				onpointerup={() => (scrubbing = false)}
				onpointercancel={() => (scrubbing = false)}
			>
				<span class="done" style:transform="scaleX({front / total})"></span>
				{#each marks as m, i (i)}
					<span class="tick" class:major={m > 28 || m % 7 === 0} style:left="{(m / total) * 100}%"
					></span>
				{/each}
				<span class="here" style:left="{(ride / total) * 100}%"></span>
			</div>
			{#if mode === 'manual' || mode === 'overview' || mode === 'finale'}
				<button onclick={() => study?.rideBack()}>Ride back</button>
				<button onclick={() => study?.follow()}
					>{front < total - 0.01 ? 'Ride along again' : 'See it whole'}</button
				>
			{/if}
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
	form {
		position: absolute;
		left: 6vw;
		top: 5vh;
		width: min(46rem, 80vw);
	}
	textarea {
		display: block;
		width: 100%;
		max-height: 30vh;
		field-sizing: content;
		resize: none;
		overflow-y: auto;
		background: none;
		border: none;
		border-bottom: 1px solid rgb(232 226 214 / 0.2);
		color: var(--bone);
		font: italic 300 clamp(1.2rem, 1.8vw, 2rem) / 1.3 var(--serif);
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
		gap: 1.2rem;
		margin-top: 0.6rem;
		font-size: 0.85rem;
		color: var(--bone);
		opacity: 0.7;
	}
	.options label {
		display: flex;
		align-items: center;
		gap: 0.45rem;
	}
	.options select,
	.options input {
		background: none;
		color: var(--bone);
		border: none;
		border-bottom: 1px solid rgb(232 226 214 / 0.25);
		font: inherit;
		padding: 0.1rem 0.2rem;
	}
	.options input {
		width: 6.5rem;
	}
	.options option {
		background: #101010;
	}
	.options .size {
		opacity: 0.6;
	}
	.caption,
	.note {
		position: absolute;
		left: 6vw;
		bottom: calc(5vh + 2.5rem);
		max-width: 62ch;
		margin: 0;
		font-weight: 300;
		line-height: 1.45;
		color: var(--bone);
		opacity: 0.75;
		pointer-events: none;
	}
	.aside {
		position: absolute;
		right: 6vw;
		top: 5vh;
		margin: 0;
		font-size: 0.85rem;
		color: var(--bone);
		opacity: 0.5;
	}
	.journey {
		position: absolute;
		left: 6vw;
		right: 6vw;
		bottom: 5vh;
		display: flex;
		align-items: center;
		gap: 1.5rem;
	}
	.track {
		position: relative;
		flex: 1;
		height: 1.6rem;
		cursor: ew-resize;
		touch-action: none;
	}
	.track:focus-visible {
		outline: 1px solid var(--ember);
		outline-offset: 4px;
	}
	.track::before {
		content: '';
		position: absolute;
		left: 0;
		right: 0;
		top: 50%;
		height: 1px;
		background: rgb(232 226 214 / 0.15);
	}
	.done {
		position: absolute;
		left: 0;
		right: 0;
		top: 50%;
		height: 1px;
		background: rgb(232 226 214 / 0.55);
		transform-origin: left;
	}
	.tick {
		position: absolute;
		top: calc(50% - 3px);
		width: 1px;
		height: 6px;
		background: rgb(232 226 214 / 0.25);
	}
	.tick.major {
		top: calc(50% - 6px);
		height: 12px;
	}
	.here {
		position: absolute;
		top: 50%;
		width: 9px;
		height: 9px;
		margin: -4.5px 0 0 -4.5px;
		border-radius: 50%;
		background: var(--ember);
	}
	button {
		background: none;
		border: 1px solid rgb(232 226 214 / 0.3);
		color: var(--bone);
		font: inherit;
		font-size: 0.9rem;
		padding: 0.3rem 0.8rem;
		cursor: pointer;
	}
	button:focus-visible {
		outline: 1px solid var(--ember);
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
</style>
