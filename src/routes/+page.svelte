<script lang="ts">
	// The landing: the threads piece on the one-file turbo model (Fast: a sketch, then the picture). Nothing runs
	// until the visitor starts it. The prompt it opens with plays a recording of a real run (no download, see
	// recording.ts); any other prompt downloads the model and runs live. The prompt can be changed at any time (Enter).
	// ?record (in development): run the opening prompt live and keep the run, for research/scripts/record_threads.mjs.
	import { asset } from '$app/paths';
	import { page } from '$app/state';
	import { ensureFonts } from '$lib/engine/text';
	import type { GPU } from '$lib/engine/gpu';
	import { labGPU } from '$lib/lab/shared';
	import { TURBO, turboPainter, turboReader, type TurboProgress } from '$lib/landing/turbo';
	import { load, pack, type Recording } from '$lib/viz/recording';
	import { Threads, type ThreadsStatus } from '$lib/viz/threads';
	import Journey from '$lib/ui/Journey.svelte';
	import Status from '$lib/ui/Status.svelte';
	import NeedsComputer from '$lib/ui/NeedsComputer.svelte';
	import { mayLoad } from '$lib/lab/device';
	import { attachOrbit } from '$lib/ui/gestures';
	import Play from '@lucide/svelte/icons/play';
	import CirclePlay from '@lucide/svelte/icons/circle-play';
	import HardDriveDownload from '@lucide/svelte/icons/hard-drive-download';
	import CornerDownLeft from '@lucide/svelte/icons/corner-down-left';

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
	let blocked = $state<{ why: string; go: () => void } | null>(null); // a live run may not fit here
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
	let paused = $state(false);
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
		const detach = attachOrbit(c, () => study);
		return () => {
			detach();
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
					for (const [name, f] of Object.entries(await pack(rec))) {
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
				s.paused = paused = false;
				await s.play(rec);
				return;
			}
			if (!live) {
				// a live run keeps about 2.1 GB on a phone's GPU (2.9 on a computer's): on a phone, ask first
				await mayLoad(2.1, 256, (why, go) => (blocked = { why, go }));
				blocked = null;
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
			s.paused = paused = false;
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

	function togglePause() {
		if (!study) return;
		study.paused = !study.paused;
		paused = study.paused;
	}

	function onkeydown(e: KeyboardEvent) {
		const tag = (e.target as HTMLElement)?.tagName;
		if (!study || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'BUTTON') return;
		if (e.key === ' ') {
			togglePause();
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
		aria-label="Your words as threads through the reader's layers, then the painter; drag to turn, scroll or pinch to zoom"
	></canvas>

	<form onsubmit={submit} class="ask" class:start={stage === 'intro'}>
		{#if stage === 'intro'}
			<p class="lede">
				Type something. A language model reads it, a diffusion model paints it, here in this tab,
				and you watch every layer do it.
			</p>
		{/if}
		<textarea
			class="prompt"
			bind:value={prompt}
			onkeydown={promptKey}
			rows="1"
			aria-label="What to paint; Enter to begin, Shift+Enter for a new line"
			spellcheck="false"></textarea>
		{#if stage === 'intro'}
			<div class="go">
				<button type="submit" class="btn primary lg" disabled={!mounted || !prompt.trim()}>
					<Play />Read and paint
				</button>
				<p class="hint">
					{#if recorded}
						<CirclePlay />
						<span
							>This prompt plays a recording of a real run, at once. Change the words to run the
							models here instead: about 1.2 GB, downloaded once and kept by your browser.</span
						>
					{:else}
						<HardDriveDownload />
						<span
							>The models run here, on your graphics card. They are about 1.2 GB, downloaded once
							and kept by your browser.</span
						>
					{/if}
				</p>
			</div>
		{:else if stage === 'journey'}
			{#if prompt.trim() && prompt.trim() !== shown}
				<p class="hint under">
					<CornerDownLeft />
					<span
						>Enter to read and paint this{live
							? ''
							: ', live: the models download first (about 1.2 GB)'}.</span
					>
				</p>
			{:else if playing}
				<p class="hint under">
					<CirclePlay /><span
						>A recording of a real run. Change the words to run it live, here.</span
					>
				</p>
			{/if}
		{/if}
	</form>

	{#if blocked}
		<NeedsComputer
			why={blocked.why}
			ontry={blocked.go}
			recorded={() => {
				blocked = null;
				prompt = RECORDED.prompt;
				begin();
			}}
		/>
	{:else if error}
		<Status text={error} error />
	{:else if stage === 'loading'}
		<Status text={loading} {progress} />
	{:else if stage === 'journey'}
		<Journey
			caption={status?.caption ?? ''}
			{front}
			{ride}
			{total}
			{marks}
			{mode}
			{done}
			{paused}
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
		--prompt-size: clamp(1.15rem, 1.8vw, 1.6rem);
		transition:
			top 0.8s ease,
			transform 0.8s ease;
	}
	.ask.start {
		top: 50%;
		width: min(50rem, calc(100% - 2 * var(--gutter)));
		transform: translateY(-55%);
		--prompt-size: clamp(1.9rem, 4.4vw, 3.5rem);
	}
	.lede {
		margin: 0 0 1.6rem;
		max-width: 34rem;
		color: var(--bone-2);
		font: 300 clamp(1.1rem, 1.5vw, 1.3rem) / 1.5 var(--serif);
		text-wrap: pretty;
	}
	.go {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		gap: 1rem 1.5rem;
		margin-top: 1.8rem;
	}
	.hint {
		display: flex;
		gap: 0.55rem;
		margin: 0;
		max-width: 27rem;
	}
	.hint :global(svg) {
		flex: none;
		width: 1rem;
		height: 1rem;
		margin-top: 0.12rem;
		color: var(--bone-3);
	}
	.go .hint {
		padding-top: 0.2rem;
	}
	.hint.under {
		margin-top: 0.6rem;
		max-width: none;
	}
	@media (max-width: 720px) {
		.ask.start {
			top: auto;
			bottom: max(1.5rem, env(safe-area-inset-bottom));
			transform: none;
		}
		.ask.start .lede {
			margin-bottom: 1.2rem;
		}
		.go .btn {
			width: 100%;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.ask {
			transition: none;
		}
	}
</style>
