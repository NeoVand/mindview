<script lang="ts">
	// A prototype of the landing, drawn from a recorded Fast run (static/_dev/landing/<run>/, from
	// research/scripts/capture_fast.py and export_landing_capture.py). ?layout=line|ring, ?t= a time in seconds,
	// ?still hides the controls (for screenshots), ?run= another recorded run.
	import { asset } from '$app/paths';
	import { page } from '$app/state';
	import { loadRun, type LandingRun } from '$lib/landing/run';
	import { DURATION, drawScene, type Layout } from '$lib/landing/scene';

	const q = page.url.searchParams;
	let layout = $state<Layout>(q.get('layout') === 'ring' ? 'ring' : 'line');
	let t = $state(Number(q.get('t') ?? 0));
	let playing = $state(!q.has('t'));
	const still = q.has('still');
	let run = $state<LandingRun | null>(null);
	let error = $state('');

	function mount(canvas: HTMLCanvasElement) {
		let frame = 0,
			last = performance.now(),
			alive = true;
		const ctx = canvas.getContext('2d')!;
		(async () => {
			await document.fonts.ready;
			run = await loadRun(asset(`/_dev/landing/${q.get('run') ?? 'bonsai'}`));
			document.body.dataset.ready = '1';
		})().catch((e) => (error = e instanceof Error ? e.message : String(e)));
		const tick = (now: number) => {
			if (!alive) return;
			const dt = (now - last) / 1000;
			last = now;
			if (playing && run) t = (t + dt) % DURATION;
			const dpr = devicePixelRatio;
			const w = canvas.clientWidth,
				h = canvas.clientHeight;
			if (canvas.width !== Math.round(w * dpr)) canvas.width = Math.round(w * dpr);
			if (canvas.height !== Math.round(h * dpr)) canvas.height = Math.round(h * dpr);
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			if (run) drawScene(ctx, run, t, layout, w, h);
			frame = requestAnimationFrame(tick);
		};
		frame = requestAnimationFrame(tick);
		return () => {
			alive = false;
			cancelAnimationFrame(frame);
		};
	}
</script>

<svelte:head><title>Landing prototype</title></svelte:head>

<main>
	<canvas {@attach mount}></canvas>
	{#if error}<p class="error">{error}</p>{/if}
	{#if !still}
		<div class="controls">
			<button type="button" onclick={() => (playing = !playing)}
				>{playing ? 'Pause' : 'Play'}</button
			>
			<input
				type="range"
				min="0"
				max={DURATION}
				step="0.01"
				bind:value={t}
				oninput={() => (playing = false)}
				aria-label="Time"
			/>
			<span>{t.toFixed(1)} s</span>
			<button type="button" onclick={() => (layout = layout === 'line' ? 'ring' : 'line')}
				>{layout === 'line' ? 'Ring' : 'Line'}</button
			>
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
		width: 100%;
		height: 100%;
		display: block;
	}
	.controls {
		position: absolute;
		top: 12px;
		right: 12px;
		display: flex;
		gap: 8px;
		align-items: center;
		font-size: 0.8rem;
		color: var(--ash);
	}
	.controls input {
		width: 320px;
	}
	.error {
		position: absolute;
		top: 1rem;
		left: 1rem;
		color: var(--ember);
	}
</style>
