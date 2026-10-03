<script lang="ts">
	import { labGPU, labModel, labPainter, releasePainting } from '$lib/lab/shared';
	import { Threads, type ThreadsStatus } from '$lib/viz/threads';
	import Journey from '$lib/ui/Journey.svelte';
	import Status from '$lib/ui/Status.svelte';
	import NeedsComputer from '$lib/ui/NeedsComputer.svelte';
	import { mayLoad } from '$lib/lab/device';
	import { attachOrbit } from '$lib/ui/gestures';
	import Dices from '@lucide/svelte/icons/dices';

	let study: Threads | undefined;
	let loading = $state('Starting the graphics card');
	let progress = $state(0);
	let ready = $state(false);
	let error = $state<string | null>(null);
	let blocked = $state<{ why: string; go: () => void } | null>(null); // the models may not fit here
	let prompt = $state('a bonsai tree made of glowing circuitry in a dark museum, volumetric light');
	let status = $state<ThreadsStatus | null>(null);
	let painterNote = $state('');
	// the journey timeline: how far the computation has got, and where the camera is
	let front = $state(0);
	let ride = $state(0);
	let total = $state(28);
	let marks = $state<number[]>([]);
	let mode = $state('live');
	let done = $state(false);
	let paused = $state(false);
	// painting options (they apply to the next painting; 'Paint again' repaints with them)
	let steps = $state(4);
	let seed = $state(7);
	let stepChoices = $state([1, 2, 3, 4, 6, 8, 12]);
	let hasPainter = $state(false);

	function togglePause() {
		if (!study) return;
		study.paused = !study.paused;
		paused = study.paused;
	}

	function mount(canvas: HTMLCanvasElement) {
		let alive = true;
		const detach = attachOrbit(canvas, () => study);

		(async () => {
			// the labs keep about 3 GB on the GPU, the reader's per-layer states in one buffer of 521 MB: on a phone, ask first
			await mayLoad(3, 521, (why, go) => (blocked = { why, go }));
			blocked = null;
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
				done = s.done ?? done;
			});
			if (import.meta.env.DEV) (window as unknown as { threads: Threads }).threads = study;
			ready = true;
			// the painter (1.1 GB) downloads while the reader runs; the journey continues into it when it is ready
			painterNote = 'The painter is downloading';
			study.painterComing = true;
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
				.catch((e) => {
					painterNote = `The painter could not load: ${e instanceof Error ? e.message : e}`;
					if (study) study.painterComing = false;
				});
			await study.read(prompt);
		})().catch((e) => (error = e instanceof Error ? e.message : String(e)));

		return () => {
			alive = false;
			detach();
			study?.destroy();
		};
	}

	function apply() {
		if (study) study.options = { steps, seed };
	}

	function submit(e?: SubmitEvent) {
		e?.preventDefault();
		apply();
		if (study && prompt.trim()) {
			study.paused = paused = false;
			study.read(prompt.trim());
		}
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
			togglePause();
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
		aria-label="Your words as threads travelling through the model's 28 layers; drag to turn, scroll or pinch to zoom"
	></canvas>
	{#if blocked}
		<NeedsComputer why={blocked.why} ontry={blocked.go} />
	{:else if error}
		<Status text={error} error />
	{:else if !ready}
		<Status text={loading} {progress} />
	{:else}
		<form onsubmit={submit} class="ask">
			<textarea
				class="prompt"
				bind:value={prompt}
				onkeydown={promptKey}
				rows="1"
				aria-label="Type something for the model to read and paint; Enter to begin, Shift+Enter for a new line"
				spellcheck="false"></textarea>
			<div class="options">
				<label class="field">
					Passes
					<select bind:value={steps} onchange={repaint} disabled={!hasPainter}>
						{#each stepChoices as n (n)}
							<option value={n}>{n}</option>
						{/each}
					</select>
				</label>
				<label class="field">
					Seed
					<input
						type="number"
						bind:value={seed}
						min="0"
						onchange={repaint}
						disabled={!hasPainter}
					/>
				</label>
				<button
					type="button"
					class="btn icon"
					onclick={newSeed}
					disabled={!hasPainter}
					aria-label="Paint again from a new seed"
					title="Paint again from a new seed"><Dices /></button
				>
				<span class="hint">512 × 512 pixels</span>
			</div>
		</form>
		<Journey
			caption={status?.caption ?? ''}
			{front}
			{ride}
			{total}
			{marks}
			{mode}
			{done}
			{paused}
			major={(m) => m > 28 || m % 7 === 0}
			onscrub={(at) => study?.rideTo(at)}
			onpause={togglePause}
			onfollow={() => study?.follow()}
			onrideback={() => study?.rideBack()}
			onpicture={() => study?.showPicture()}
		>
			{#snippet note()}
				{#if painterNote}<p class="hint">{painterNote}</p>{/if}
			{/snippet}
		</Journey>
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
	.ask {
		position: absolute;
		left: var(--gutter);
		top: var(--top);
		width: min(44rem, calc(100% - 2 * var(--gutter)));
	}
	.options {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.6rem 1rem;
		margin-top: 0.75rem;
	}
	.options .field input {
		width: 5.5rem;
	}
</style>
