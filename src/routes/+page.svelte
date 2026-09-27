<script lang="ts">
	// The landing: the threads piece on the one-file turbo model (Fast: a sketch, then the picture). Nothing runs
	// until the visitor starts it. The prompt it opens with plays a recording of a real run (no download, see
	// recording.ts); any other prompt downloads the model and runs live. The prompt can be changed at any time (Enter).
	// ?record (in development): run the opening prompt live and keep the run, for research/scripts/record_threads.mjs.
	import { asset, resolve } from '$app/paths';
	import { page } from '$app/state';
	import { ensureFonts } from '$lib/engine/text';
	import type { GPU } from '$lib/engine/gpu';
	import { labGPU } from '$lib/lab/shared';
	import { TURBO, turboPainter, turboReader, type TurboProgress } from '$lib/landing/turbo';
	import { load, pack, type Recording } from '$lib/viz/recording';
	import { Threads, type ThreadsStatus } from '$lib/viz/threads';

	const RECORDED = {
		prompt: 'a bonsai tree made of glowing circuitry in a dark museum, volumetric light',
		base: asset('/recordings/bonsai-museum')
	};
	const record = import.meta.env.DEV && page.url.searchParams.has('record');

	let study: Threads | undefined;
	let gpu: GPU | undefined;
	let canvas: HTMLCanvasElement | undefined;
	let recording: Promise<Recording> | undefined;
	let stage = $state<'intro' | 'loading' | 'journey'>('intro');
	let prompt = $state(RECORDED.prompt);
	let playing = $state(false); // the journey on screen is the recording
	let shown = $state(''); // the prompt on screen
	let live = $state(false); // the reader is loaded
	let error = $state<string | null>(null);
	let loading = $state('');
	let progress = $state(0);
	let painterNote = $state('');
	let status = $state<ThreadsStatus | null>(null);
	let front = $state(0);
	let ride = $state(0);
	let total = $state(9);
	let marks = $state<number[]>([]);
	let mode = $state('live');
	let done = $state(false);
	let scrubbing = false;
	let mounted = $state(false); // the canvas is in the page (the button waits for it)
	const mb = (x: number) => Math.round(x).toLocaleString();
	const recorded = $derived(!record && prompt.trim() === RECORDED.prompt);

	function onProgress(p: TurboProgress) {
		if (p.stage === 'reader') {
			progress = p.megabytes / p.total;
			loading = `Downloading the reader: ${mb(p.megabytes)} of ${mb(p.total)} MB. Your browser keeps the model after the first time.`;
		} else
			painterNote =
				p.fraction < 1 ? `The painter is downloading: ${mb(p.megabytes)} of ${mb(p.total)} MB` : '';
	}

	function onStatus(s: ThreadsStatus) {
		if (s.caption !== status?.caption || s.busy !== status?.busy) status = s;
		front = s.front ?? front;
		ride = s.ride ?? ride;
		total = s.total ?? total;
		marks = s.marks ?? marks;
		mode = s.mode ?? mode;
		done = s.done ?? done;
	}

	function mount(c: HTMLCanvasElement) {
		canvas = c;
		mounted = true;
		let drag: { x: number; y: number; id: number } | null = null;
		const down = (e: PointerEvent) => {
			drag = { x: e.clientX, y: e.clientY, id: e.pointerId };
			c.setPointerCapture(e.pointerId);
		};
		const move = (e: PointerEvent) => {
			if (!drag || !study || e.pointerId !== drag.id) return;
			study.orbit(e.clientX - drag.x, e.clientY - drag.y);
			drag.x = e.clientX;
			drag.y = e.clientY;
		};
		const up = () => (drag = null);
		const wheel = (e: WheelEvent) => {
			if (!study) return;
			e.preventDefault();
			study.zoom(Math.exp(e.deltaY * 0.0015));
		};
		c.addEventListener('pointerdown', down);
		c.addEventListener('pointermove', move);
		c.addEventListener('pointerup', up);
		c.addEventListener('pointercancel', up);
		c.addEventListener('wheel', wheel, { passive: false });
		return () => {
			c.removeEventListener('pointerdown', down);
			c.removeEventListener('pointermove', move);
			c.removeEventListener('pointerup', up);
			c.removeEventListener('pointercancel', up);
			c.removeEventListener('wheel', wheel);
			study?.destroy();
			study = undefined;
		};
	}

	/** The piece, made once (without the reader until a live run needs it). */
	async function piece(): Promise<Threads> {
		gpu ??= await labGPU(canvas!);
		if (!study) {
			study = new Threads(gpu, undefined, onStatus, TURBO);
			study.record = record;
			if (import.meta.env.DEV) {
				const w = window as unknown as { threads: Threads; recordingFiles: () => unknown };
				w.threads = study;
				// the recording's files, as base64, for record_threads.mjs
				w.recordingFiles = async () => {
					const rec = study?.recording;
					if (!rec) return null;
					const out: Record<string, string> = {};
					for (const [name, f] of Object.entries(pack(rec))) {
						const bytes = new Uint8Array(
							typeof f === 'string' ? new TextEncoder().encode(f) : await f.arrayBuffer()
						);
						let bin = '';
						for (let i = 0; i < bytes.length; i += 0x8000)
							bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
						out[name] = btoa(bin);
					}
					return out;
				};
			}
		}
		return study;
	}

	async function begin() {
		if (!canvas) return;
		const text = prompt.trim();
		error = null;
		try {
			if (recorded) {
				// the recording: no model needed
				if (stage === 'intro') {
					stage = 'loading';
					loading = 'Loading the recording';
				}
				recording ??= load(RECORDED.base).catch((e) => {
					recording = undefined;
					throw e;
				});
				const [s, rec] = await Promise.all([piece(), recording, ensureFonts()]);
				stage = 'journey';
				playing = true;
				shown = text;
				await s.play(rec);
				return;
			}
			const s = await piece();
			if (!live) {
				stage = 'loading';
				loading = 'Starting the reader';
				s.attachReader(await turboReader(gpu!.device, onProgress));
				live = true;
				painterNote = 'The painter is downloading';
				s.painterComing = true;
				turboPainter(gpu!.device, onProgress)
					.then((p) => {
						painterNote = '';
						study?.attachPainter(p);
					})
					.catch((e) => {
						painterNote = `The painter could not load: ${e instanceof Error ? e.message : e}`;
						s.painterComing = false;
					});
			}
			stage = 'journey';
			playing = false;
			shown = text;
			await s.read(text);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	function submit(e?: SubmitEvent) {
		e?.preventDefault();
		if (prompt.trim()) begin();
	}

	/** Enter starts; Shift+Enter starts a new line. */
	function promptKey(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submit();
		}
	}

	function scrub(e: PointerEvent) {
		const box = (e.currentTarget as HTMLElement).getBoundingClientRect();
		study?.rideTo(((e.clientX - box.left) / box.width) * total);
	}

	function onkeydown(e: KeyboardEvent) {
		const tag = (e.target as HTMLElement)?.tagName;
		if (!study || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'BUTTON') return;
		if (e.key === ' ') {
			study.paused = !study.paused;
			e.preventDefault();
		}
	}
</script>

<svelte:head>
	<title>mindview</title>
	<meta
		name="description"
		content="A ternary language model reads your words and a ternary diffusion model paints them, in your browser, and you watch every layer compute."
	/>
</svelte:head>
<svelte:window {onkeydown} />

<main class:intro={stage === 'intro'}>
	<canvas
		{@attach mount}
		aria-label="Your words as threads through the reader's layers, then the painter; drag to turn, scroll to zoom"
	></canvas>
	<nav class="elsewhere" aria-label="Elsewhere">
		<a href={resolve('/paint')}>Paint with it</a>
		<a href={resolve('/lab')}>The labs: the models live, every number</a>
	</nav>

	<form onsubmit={submit} class:start={stage === 'intro'}>
		{#if stage === 'intro'}
			<p class="lede">
				Type something. A language model reads it, a diffusion model paints it, here in this tab,
				and you watch every layer do it.
			</p>
		{/if}
		<textarea
			bind:value={prompt}
			onkeydown={promptKey}
			rows="1"
			aria-label="What to paint; Enter to begin, Shift+Enter for a new line"
			spellcheck="false"></textarea>
		{#if stage === 'intro'}
			<div class="go">
				<button type="submit" disabled={!mounted || !prompt.trim()}>Read and paint</button>
				<span class="size"
					>{recorded
						? 'This prompt plays a recording of a real run. Change the words to run the models here, on your graphics card: about 1.2 GB, downloaded once and kept by your browser.'
						: 'The models are about 1.2 GB, downloaded once and kept by your browser. They run here, on your graphics card.'}</span
				>
			</div>
		{:else if stage === 'journey'}
			{#if prompt.trim() && prompt.trim() !== shown}
				<p class="under">
					Enter to read and paint this{live
						? ''
						: ', live: the models download first (about 1.2 GB)'}.
				</p>
			{:else if painterNote}
				<p class="under">{painterNote}</p>
			{:else if playing}
				<p class="under">A recording of a real run. Change the words to run it live, here.</p>
			{/if}
		{/if}
	</form>

	{#if error}
		<p class="note" role="alert">{error}</p>
	{:else if stage === 'loading'}
		<div class="note">
			<p>{loading}</p>
			<div class="bar"><span style:transform="scaleX({progress})"></span></div>
		</div>
	{:else if stage === 'journey'}
		<p class="caption" aria-live="polite">{status?.caption ?? ''}</p>

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
					<span class="tick" style:left="{(m / total) * 100}%"></span>
				{/each}
				<span class="here" style:left="{(ride / total) * 100}%"></span>
			</div>
			{#if mode !== 'live'}
				{#if front < total - 0.01}
					<button onclick={() => study?.follow()}>Ride along again</button>
				{:else}
					{#if mode !== 'rewind'}<button onclick={() => study?.rideBack()}>Ride back</button>{/if}
					{#if mode !== 'overview'}<button onclick={() => study?.follow()}>See it whole</button
						>{/if}
					{#if mode !== 'finale' && done}<button onclick={() => study?.showPicture()}
							>The picture</button
						>{/if}
				{/if}
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
		width: min(46rem, 88vw);
		transition:
			top 0.8s ease,
			transform 0.8s ease;
	}
	form.start {
		top: 50%;
		transform: translateY(-60%);
	}
	.lede {
		margin: 0 0 1.4rem;
		max-width: 34rem;
		font-weight: 300;
		font-size: clamp(1rem, 1.3vw, 1.15rem);
		line-height: 1.5;
		color: var(--bone);
		opacity: 0.7;
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
	form.start textarea {
		font-size: clamp(1.6rem, 3.2vw, 3rem);
	}
	textarea:focus-visible {
		border-bottom-color: var(--ember);
	}
	.go {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 1.2rem 1.6rem;
		margin-top: 1.6rem;
	}
	.go button {
		font-size: 1.05rem;
		padding: 0.55rem 1.3rem;
		border-color: var(--ember);
	}
	.size {
		max-width: 26rem;
		font-size: 0.85rem;
		line-height: 1.45;
		color: var(--bone);
		opacity: 0.5;
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
	.under {
		margin: 0.6rem 0 0;
		font-size: 0.85rem;
		line-height: 1.45;
		color: var(--bone);
		opacity: 0.5;
	}
	.elsewhere {
		position: absolute;
		top: 1.1rem;
		right: 1.4rem;
		z-index: 2;
		display: flex;
		gap: 1.1rem;
	}
	.elsewhere a {
		font-size: 0.82rem;
		letter-spacing: 0.01em;
		color: var(--bone);
		opacity: 0.55;
		text-decoration: none;
		border-bottom: 1px solid transparent;
	}
	.elsewhere a:hover,
	.elsewhere a:focus-visible {
		opacity: 1;
		border-bottom-color: var(--ember);
		outline: none;
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
	button:disabled {
		opacity: 0.4;
		cursor: default;
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
	@media (prefers-reduced-motion: reduce) {
		form {
			transition: none;
		}
	}
</style>
