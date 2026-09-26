<script lang="ts">
	import { readerUrl } from '$lib/models';
	import { initGPU } from '$lib/engine/gpu';
	import { ensureFonts } from '$lib/engine/text';
	import { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import { Assembly, type AssemblyStatus } from '$lib/viz/assembly';

	let study: Assembly | undefined;
	let loading = $state('Starting the graphics card');
	let progress = $state(0);
	let ready = $state(false);
	let error = $state<string | null>(null);
	let prompt = $state('a bonsai tree made of glowing circuitry in a dark museum, volumetric light');
	let status = $state<AssemblyStatus | null>(null);

	function mount(canvas: HTMLCanvasElement) {
		let alive = true;
		(async () => {
			const gpu = await initGPU(canvas);
			loading = 'Loading Ternary Bonsai 1.7B (460 MB)';
			const [llm] = await Promise.all([
				BonsaiLLM.load(gpu.device, readerUrl(), (f) => (progress = f)),
				ensureFonts()
			]);
			if (!alive) return;
			study = new Assembly(gpu, llm, (s) => {
				if (s.caption !== status?.caption || s.busy !== status?.busy) status = s;
			});
			if (import.meta.env.DEV) (window as unknown as { assembly: Assembly }).assembly = study;
			ready = true;
			await study.read(prompt);
		})().catch((e) => (error = e instanceof Error ? e.message : String(e)));
		return () => {
			alive = false;
			study?.destroy();
		};
	}

	function submit(e: SubmitEvent) {
		e.preventDefault();
		if (study && prompt.trim()) study.read(prompt.trim());
	}

	function onkeydown(e: KeyboardEvent) {
		if (!study || (e.target as HTMLElement)?.tagName === 'INPUT') return;
		if (e.key === ' ') {
			study.paused = !study.paused;
			e.preventDefault();
		} else if (e.key === 'ArrowRight') study.seekLayer(1);
		else if (e.key === 'ArrowLeft') study.seekLayer(-1);
	}
</script>

<svelte:head><title>Meaning assembling</title></svelte:head>
<svelte:window {onkeydown} />

<main>
	<canvas
		{@attach mount}
		aria-label="The words of your prompt, placed by what the model makes of them"
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
			<input
				bind:value={prompt}
				aria-label="Type something for the model to read"
				spellcheck="false"
			/>
		</form>
		<p class="caption" aria-live="polite">
			{status?.busy ? 'Reading your words through all 28 layers' : (status?.caption ?? '')}
		</p>
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
	}
	form {
		position: absolute;
		left: 6vw;
		top: 5vh;
		width: min(46rem, 80vw);
	}
	input {
		width: 100%;
		background: none;
		border: none;
		border-bottom: 1px solid rgb(232 226 214 / 0.2);
		color: var(--bone);
		font: italic 300 clamp(1.2rem, 1.8vw, 2rem) / 1.3 var(--serif);
		padding: 0.3rem 0;
		outline: none;
	}
	input:focus-visible {
		border-bottom-color: var(--ember);
	}
	.caption,
	.note {
		position: absolute;
		left: 6vw;
		bottom: 5vh;
		margin: 0;
		font-weight: 300;
		color: var(--bone);
		opacity: 0.7;
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
