<script lang="ts">
	import { asset, resolve } from '$app/paths';
	import { Engine, type EngineStatus } from '$lib/engine';

	let engine: Engine | undefined;

	let ready = $state(false);
	let progress = $state(0);
	let loading = $state('Starting');
	let error = $state<string | null>(null);
	let caption = $state('');
	let act = $state<EngineStatus['act']>(0);
	let prompt = $state('');
	let paused = $state(false);
	let hint = $state(true);

	function onStatus(s: EngineStatus) {
		// called every frame; only touch reactive state when something visible changed
		if (s.caption !== caption) caption = s.caption;
		if (s.act !== act) act = s.act;
		if (s.paused !== paused) paused = s.paused;
		if (s.prompt !== prompt) prompt = s.prompt;
	}

	/** Attachment: start the engine on the canvas once it is in the page, tear it down with it. */
	function mountEngine(canvas: HTMLCanvasElement) {
		engine = new Engine(canvas, onStatus);
		if (import.meta.env.DEV) (window as unknown as { mindview: Engine }).mindview = engine; // for scrubbing from devtools
		const manifest = asset('/traces/bonsai-museum-ternary/manifest.json');
		engine
			.start(manifest.replace(/\/manifest\.json$/, ''), (f, what) => {
				progress = f;
				loading = what;
			})
			.then(() => {
				ready = true;
				setTimeout(() => (hint = false), 9000);
			})
			.catch((e) => (error = e instanceof Error ? e.message : String(e)));
		return () => engine?.destroy();
	}

	function onkeydown(e: KeyboardEvent) {
		if (!engine || !ready) return;
		if (e.key === ' ') {
			engine.paused = !engine.paused;
			e.preventDefault();
		} else if (e.key === 'ArrowRight') engine.seek(engine.time + (e.shiftKey ? 15 : 3));
		else if (e.key === 'ArrowLeft') engine.seek(engine.time - (e.shiftKey ? 15 : 3));
		else if (e.key === 'c') engine.palette = 1 - engine.palette;
		else if (e.key === 'f') {
			if (document.fullscreenElement) document.exitFullscreen();
			else document.documentElement.requestFullscreen();
		} else if (e.key === 'r') engine.seek(0);
		else return;
		hint = false;
	}
</script>

<svelte:head>
	<title>mindview</title>
	<meta
		name="description"
		content="A ternary language model reads your words and a ternary painter turns them into an image, shown as it computes."
	/>
</svelte:head>

<svelte:window {onkeydown} />

<main class:ready>
	<canvas {@attach mountEngine} aria-label="The model computing, drawn as light"></canvas>
	<a class="labs" href={resolve('/lab')}>The labs: the models live, every number</a>

	{#if error}
		<p class="error" role="alert">{error}</p>
	{:else if !ready}
		<div class="loading" aria-live="polite">
			<p>{loading}</p>
			<div class="bar"><span style:transform="scaleX({progress})"></span></div>
		</div>
	{:else}
		<p class="prompt" class:intro={act === 0} class:finale={act === 4}>{prompt}</p>
		<p class="caption" aria-live="polite">{caption}{paused ? ' (paused)' : ''}</p>
		{#if hint}
			<p class="hint">
				Space pauses. The arrow keys move through time. C switches the colours. F fills the screen.
			</p>
		{/if}
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
		display: block;
	}
	p {
		margin: 0;
		position: absolute;
		pointer-events: none;
	}

	.prompt {
		left: 6vw;
		top: 6vh;
		max-width: 34ch;
		font-style: italic;
		font-weight: 300;
		font-size: clamp(1.25rem, 1.9vw, 2.1rem);
		line-height: 1.3;
		letter-spacing: 0.005em;
		color: var(--bone);
		opacity: 0.8;
		transition:
			font-size 2.4s cubic-bezier(0.2, 0.7, 0.1, 1),
			top 2.4s cubic-bezier(0.2, 0.7, 0.1, 1),
			opacity 2.4s ease;
	}
	/* the visitor's words open the piece, large, before the model starts reading them */
	.prompt.intro {
		top: 38vh;
		font-size: clamp(2rem, 4.2vw, 4.6rem);
		opacity: 1;
	}
	.prompt.finale {
		top: auto;
		bottom: 11vh;
		opacity: 0.9;
	}

	.caption {
		left: 6vw;
		bottom: 5vh;
		font-weight: 300;
		font-size: clamp(0.95rem, 1.05vw, 1.2rem);
		line-height: 1.5;
		color: var(--bone);
		opacity: 0.62;
		font-variant-numeric: oldstyle-nums proportional-nums;
	}

	.hint {
		right: 6vw;
		bottom: 5vh;
		max-width: 38ch;
		text-align: right;
		font-size: 0.9rem;
		line-height: 1.5;
		color: var(--ash);
	}

	.loading {
		position: absolute;
		left: 6vw;
		bottom: 5vh;
		width: min(28rem, 60vw);
	}
	.loading p {
		position: static;
		font-weight: 300;
		font-size: 1rem;
		color: var(--bone);
		opacity: 0.7;
		margin-bottom: 0.8rem;
	}
	.bar {
		height: 1px;
		background: rgb(232 226 214 / 0.15);
	}
	.bar span {
		display: block;
		height: 100%;
		background: var(--ember);
		transform-origin: left;
		transition: transform 0.2s linear;
	}

	.labs {
		position: absolute;
		top: 1.1rem;
		right: 1.4rem;
		z-index: 2;
		font-size: 0.82rem;
		letter-spacing: 0.01em;
		color: var(--bone);
		opacity: 0.55;
		text-decoration: none;
		border-bottom: 1px solid transparent;
	}
	.labs:hover,
	.labs:focus-visible {
		opacity: 1;
		border-bottom-color: var(--ember);
		outline: none;
	}

	.error {
		left: 6vw;
		top: 40vh;
		max-width: 46ch;
		font-size: 1.2rem;
		line-height: 1.5;
		color: var(--bone);
	}

	@media (prefers-reduced-motion: reduce) {
		.prompt {
			transition: none;
		}
	}
</style>
